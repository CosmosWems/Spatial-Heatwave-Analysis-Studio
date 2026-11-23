"""
Spatial Heatwave Engine
=======================
Adapts the Hobday et al. (2016) heatwave detection (as implemented by the
marineHeatWaves package) to gridded (spatial) meteorological data, focusing
on *land* and user-supplied study areas. Given one or more time slices of
temperature on a lat-lon grid (NetCDF / GeoTIFF / raster stack), this engine:

  * masks the domain to a user-supplied study-area shapefile boundary,
  * optionally excludes ocean/sea cells (cells with no valid data, or a
    user-supplied static land/ocean mask),
  * runs the Hobday et al. (2016) heatwave detection independently at
    every grid point, in parallel over the requested number of cores,
  * produces maps of heatwave metrics (count, intensity, duration,
    frequency, cumulative intensity, days), annual/5-year statistics,
    and spatial trend fields, and
  * supports comparative analysis between two gridded fields
    (e.g., ERA5 vs observation, or two climatological periods).
"""

from __future__ import annotations

import os
import io
import json
import datetime
import warnings
from datetime import date as _date
from collections import OrderedDict

import numpy as np
import pandas as pd

import marineHeatWaves as mhw

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

warnings.filterwarnings("ignore")


def _temperature_variable_candidates(ds):
    """Return likely temperature variables ordered by Tmax/Tmin semantics."""
    ranked = []
    aliases = {
        'tmax': ('tmax', 'tx', 'tasmax', 'maximum_temperature', 'max_temperature'),
        'tmin': ('tmin', 'tn', 'tasmin', 'minimum_temperature', 'min_temperature'),
    }
    for name, variable in ds.variables.items():
        if getattr(variable, 'ndim', 0) != 3:
            continue
        text = ' '.join(str(getattr(variable, attr, '')) for attr in ('standard_name', 'long_name'))
        haystack = f'{name} {text}'.lower().replace('-', '_').replace(' ', '_')
        role = 'other'
        for candidate_role, names in aliases.items():
            if any(token in haystack for token in names):
                role = candidate_role
                break
        ranked.append((0 if role == 'tmax' else 1 if role == 'tmin' else 2, name, role))
    return sorted(ranked)


def convert_temperature(values, unit='auto', source_units=''):
    """Convert temperature values to Celsius using explicit or automatic units.

    Works in-place on 32-bit floats so that large grids do not incur an
    extra float64 allocation. The Kelvin auto-check is estimated from a
    decimated sample of the values (never a full float64 copy).
    """
    values = np.asarray(values, dtype=np.float32)
    requested = str(unit or 'auto').strip().lower()
    units_text = str(source_units or '').lower()
    is_kelvin = requested in ('k', 'kelvin', 'kelvins')
    is_celsius = requested in ('c', 'degc', 'celsius', 'degree_celsius', 'degrees_celsius')
    if requested == 'auto':
        is_kelvin = 'kelvin' in units_text or units_text.strip() in ('k', 'deg_k')
        if not is_kelvin:
            flat = values.ravel()
            step = max(1, flat.size // 5_000_000)
            sample = flat[::step]
            sample = sample[np.isfinite(sample)]
            is_kelvin = bool(sample.size and np.nanmedian(sample) > 200)
    if is_kelvin and not is_celsius:
        return values - np.float32(273.15), True
    return values, False

try:
    import netCDF4
    HAS_NETCDF4 = True
except Exception:
    HAS_NETCDF4 = False

try:
    import xarray as xr
    HAS_XARRAY = True
except Exception:
    HAS_XARRAY = False

try:
    import rasterio
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except Exception:
    HAS_GEOPANDAS = False

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ----------------------------------------------------------------------------
#  Grid container
# ----------------------------------------------------------------------------

class HeatwaveGrid:
    """Simple container for a 3D temperature grid (time, lat, lon)."""

    def __init__(self, values, lats, lons, times, variable='temp'):
        self.values = np.asarray(values, dtype=np.float32)   # (nt, nlat, nlon)
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.times = pd.DatetimeIndex(pd.to_datetime(times))
        self.variable = variable

        if self.values.ndim != 3:
            raise ValueError('Grid values must be 3-dimensional (time, lat, lon).')
        nt, nlat, nlon = self.values.shape
        if len(self.lats) != nlat:
            raise ValueError('Latitude axis length does not match grid.')
        if len(self.lons) != nlon:
            raise ValueError('Longitude axis length does not match grid.')
        if len(self.times) != nt:
            raise ValueError('Time axis length does not match grid.')

        # Sort so that lats ascend and lons ascend (raster-style index mapping)
        self._sort_axes()

    def _sort_axes(self):
        if self.lats[0] > self.lats[-1]:
            self.values = self.values[:, ::-1, :]
            self.lats = self.lats[::-1]
        if self.lons[0] > self.lons[-1]:
            self.values = self.values[:, :, ::-1]
            self.lons = self.lons[::-1]

    def copy(self):
        g = HeatwaveGrid(
            self.values.copy(), self.lats.copy(), self.lons.copy(),
            self.times.copy(), self.variable)
        g.source_units = getattr(self, 'source_units', '')
        g.converted_from_kelvin = getattr(self, 'converted_from_kelvin', False)
        return g

    @property
    def shape(self):
        return self.values.shape

    @property
    def ntime(self):
        return self.values.shape[0]

    @property
    def nlat(self):
        return self.values.shape[1]

    @property
    def nlon(self):
        return self.values.shape[2]

    @property
    def dlon(self):
        if self.nlon > 1:
            return float(abs(self.lons[1] - self.lons[0]))
        return 1.0

    @property
    def dlat(self):
        if self.nlat > 1:
            return float(abs(self.lats[1] - self.lats[0]))
        return 1.0

    def t_ordinal(self):
        return np.array([d.toordinal() for d in self.times])

    # ------------------------------ I/O ------------------------------
    @staticmethod
    def _read_var_chunked(var, max_bytes=2 * 1024 ** 3):
        """Read a NetCDF variable into float32 in bounded time-chunks.

        Non-NaN fill/missing markers and scale_factor/add_offset are handled
        per chunk, so arbitrarily large (time, lat, lon) variables can be
        loaded without a giant float64 allocation. ``max_bytes`` caps the
        working memory used by each chunk read.
        """
        ntime, nlat, nlon = var.shape
        # allocate in caller-controlled chunks
        rows_bytes = nlat * nlon * np.dtype(np.float32).itemsize
        chunk = max(1, int(max_bytes / max(1, rows_bytes)))
        chunk = min(chunk, ntime)
        result = np.empty((ntime, nlat, nlon), dtype=np.float32)
        scale = getattr(var, 'scale_factor', None)
        offset = getattr(var, 'add_offset', None)
        fill = getattr(var, '_FillValue', None)
        missing = getattr(var, 'missing_value', None)
        markers = [m for m in (fill, missing) if m is not None]

        for t0 in range(0, ntime, chunk):
            t1 = min(ntime, t0 + chunk)
            block = var[t0:t1]
            if np.ma.isMaskedArray(block):
                mask = np.ma.getmaskarray(block)
                data = np.ma.filled(block, np.nan).astype(np.float32)
                if mask.any():
                    data = data.copy()
                    data[mask] = np.nan
            else:
                data = np.asarray(block, dtype=np.float32).copy()
            if scale is not None:
                data *= np.float32(scale)
            if offset is not None:
                data += np.float32(offset)
            for marker in markers:
                mv = np.float32(marker)
                if np.isnan(mv):
                    continue
                data[data == mv] = np.nan
            result[t0:t1] = data
        return result

    @classmethod
    def from_netcdf(cls, path, var_name=None, time_name='time',
                    lat_name='lat', lon_name='lon', temperature_unit='auto'):
        """Load a 3D (time, lat, lon) temperature field from NetCDF."""
        if not HAS_NETCDF4:
            raise ImportError('netCDF4 is required to read NetCDF files. pip install netCDF4')
        ds = netCDF4.Dataset(path, 'r')
        try:
            if var_name in ('tmax', 'tmin'):
                role_candidates = [item for item in _temperature_variable_candidates(ds)
                                    if item[2] == var_name]
                if not role_candidates:
                    raise ValueError(f'No variable matching {var_name.upper()} / {"TX" if var_name == "tmax" else "TN"} was found.')
                var_name = role_candidates[0][1]
            if var_name in (None, '', 'auto'):
                candidates = _temperature_variable_candidates(ds)
                if not candidates:
                    raise ValueError('No 3D variable found in NetCDF file. Please specify variable name.')
                var_name = candidates[0][1]
            var = ds.variables[var_name]
            if var.ndim != 3:
                raise ValueError(f'Variable "{var_name}" must be 3D (time, lat, lon); got ndim={var.ndim}.')

            def find_coordinate(requested, aliases):
                names = {str(name).lower(): name for name in ds.variables}
                if requested in ds.variables:
                    return requested
                for alias in aliases:
                    if alias in names:
                        return names[alias]
                for dimension in var.dimensions:
                    if str(dimension).lower() in aliases and dimension in ds.variables:
                        return dimension
                raise ValueError(
                    f'Could not find {requested} coordinate. Expected one of: '
                    f'{", ".join(aliases)}.'
                )

            time_var = find_coordinate(time_name, ('time', 'date', 'datetime', 'valid_time'))
            lat_var = find_coordinate(lat_name, ('lat', 'latitude', 'y'))
            lon_var = find_coordinate(lon_name, ('lon', 'longitude', 'x'))

            times = ds.variables[time_var][:]
            lats = ds.variables[lat_var][:]
            lons = ds.variables[lon_var][:]
            values = cls._read_var_chunked(var)
            values, converted_from_kelvin = convert_temperature(
                values, temperature_unit, getattr(var, 'units', '')
            )

            dim_names = [str(dim).lower() for dim in var.dimensions]
            expected = [str(time_var).lower(), str(lat_var).lower(), str(lon_var).lower()]
            if all(name in dim_names for name in expected):
                values = np.transpose(values, [dim_names.index(name) for name in expected])

            # Standardize longitude to -180..180
            lons_arr = np.asarray(lons, dtype=float)
            lons_arr = np.where(lons_arr > 180, lons_arr - 360, lons_arr)
            values = np.asarray(values, dtype=np.float32)

            # Build pandas time index
            try:
                tvar = ds.variables[time_var]
                tunit = tvar.units
                calendar = getattr(tvar, 'calendar', 'standard')
                times_pd = _netcdf_times_to_datetime(times, tunit, calendar)
            except Exception:
                times_pd = pd.to_datetime(np.asarray(times, dtype=float),
                                          unit='D', origin='unix')
            grid = cls(values, lats, lons, times_pd, var_name)
            grid.source_units = getattr(var, 'units', '')
            grid.converted_from_kelvin = converted_from_kelvin
            return grid
        finally:
            ds.close()

    @staticmethod
    def list_all_variables(path):
        """Return metadata for every 3-D variable in a NetCDF file.

        Each entry contains ``name``, ``ndim``, ``shape``,
        ``standard_name``, ``long_name``, ``units`` and a suggested
        ``role`` (``'tmax'``, ``'tmin'`` or ``'other'``).
        """
        if not HAS_NETCDF4:
            raise ImportError('netCDF4 is required to read NetCDF files.')
        ds = netCDF4.Dataset(path, 'r')
        try:
            results = []
            for name, var in ds.variables.items():
                if getattr(var, 'ndim', 0) != 3:
                    continue
                text = ' '.join(str(getattr(var, attr, ''))
                                for attr in ('standard_name', 'long_name'))
                haystack = f'{name} {text}'.lower().replace('-', '_').replace(' ', '_')
                aliases = {
                    'tmax': ('tmax', 'tx', 'tasmax', 'maximum_temperature',
                             'max_temperature'),
                    'tmin': ('tmin', 'tn', 'tasmin', 'minimum_temperature',
                             'min_temperature'),
                }
                role = 'other'
                for candidate_role, tokens in aliases.items():
                    if any(tok in haystack for tok in tokens):
                        role = candidate_role
                        break
                results.append({
                    'name': name,
                    'ndim': int(var.ndim),
                    'shape': [int(s) for s in var.shape],
                    'standard_name': str(getattr(var, 'standard_name', '')),
                    'long_name': str(getattr(var, 'long_name', '')),
                    'units': str(getattr(var, 'units', '')),
                    'role': role,
                })
            return results
        finally:
            ds.close()

    @classmethod
    def from_netcdf_variable(cls, path, var_name, temperature_unit='auto'):
        """Load a single named variable from a NetCDF file as a grid."""
        return cls.from_netcdf(path, var_name=var_name,
                               temperature_unit=temperature_unit)

    @classmethod
    def from_geotiff_stack(cls, paths, variable='T2m'):
        """Load a stack of single-band GeoTIFFs (ordered by time)."""
        if not HAS_RASTERIO:
            raise ImportError('rasterio is required to read GeoTIFF files. pip install rasterio')
        stacked = []
        times = []
        for i, p in enumerate(sorted(paths)):
            with rasterio.open(p) as src:
                band = src.read(1)
                if i == 0:
                    lons = src.xy(np.zeros(src.height, dtype=int),
                                  np.arange(src.width))[0]
                    lats = [src.xy(r, 0)[1] for r in np.arange(src.height)]
                    lons = np.array(lons)
                    lats = np.array(lats)
                stacked.append(band)
            times.append(pd.to_datetime(os.path.basename(p).split('.')[0]))
        values = np.stack(stacked, axis=0)
        return cls(values, lats, lons, times, variable)

    def to_netcdf(self, path, var_name=None):
        """Write grid back to NetCDF (useful for exporting masked grids)."""
        if not HAS_NETCDF4:
            raise ImportError('netCDF4 required for NetCDF export.')
        ds = netCDF4.Dataset(path, 'w', format='NETCDF4')
        try:
            ntime, nlat, nlon = self.shape
            ds.createDimension('time', ntime)
            ds.createDimension('lat', nlat)
            ds.createDimension('lon', nlon)
            tvar = ds.createVariable('time', 'f8', ('time',))
            tvar[:] = self.t_ordinal()
            tvar.units = 'days since 0001-01-01'
            latv = ds.createVariable('lat', 'f8', ('lat',))
            lonv = ds.createVariable('lon', 'f8', ('lon',))
            latv[:] = self.lats
            lonv[:] = self.lons
            vv = ds.createVariable(var_name or self.variable, 'f8',
                                   ('time', 'lat', 'lon'),
                                   zlib=True, fill_value=np.nan)
            vv.units = 'degree_Celsius'
            vv[:] = self.values
        finally:
            ds.close()

    def add_noise_if_needed(self, max_bytes=2 * 1024 ** 3):
        """Add tiny random noise to break ties (some percentile methods choke).

        Skipped for very large grids to avoid a full-size float64 allocation;
        tie values are rare in real observations so the noise is optional.
        """
        if self.values.nbytes > max_bytes:
            return
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(self.values.shape, dtype=np.float32)
        self.values += noise * np.float32(1e-6)

    # ------------------------------ Masking ------------------------------
    def build_mask_from_latlon(self, bounds, lat_name='lat', lon_name='lon'):
        """Create a boolean mask for a simple lat/lon bounding box."""
        lat_min, lat_max, lon_min, lon_max = bounds
        mask = np.ones((self.nlat, self.nlon), dtype=bool)
        for i, lat in enumerate(self.lats):
            for j, lon in enumerate(self.lons):
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    mask[i, j] = False
        return mask

    def mask_with_shapefile(self, shp_path, mask_value=np.nan, buffer_degrees=0.0):
        """Mask grid using shapefile: cells whose centroid touches any polygon are kept.

        Uses rtree spatial index for fast lookup, with a fallback to
        ``intersects`` (instead of ``contains``) so that grid cells whose
        centroid lies on a polygon boundary are still included.  An optional
        buffer (in degrees) expands each polygon slightly before testing,
        which is useful for coarse grids where centroids may fall just
        outside the study-area boundary.
        """
        if not HAS_GEOPANDAS:
            raise ImportError('geopandas is required for shapefile masking. pip install geopandas')
        from shapely.geometry import Point
        from shapely.ops import unary_union

        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        gdf = gdf.to_crs(epsg=4326)

        # Build a single dissolved geometry for faster testing
        merged = unary_union(gdf.geometry.tolist())
        if buffer_degrees > 0:
            merged = merged.buffer(buffer_degrees)

        # Build an rtree spatial index from the merged bounds
        try:
            from rtree import index as rtree_index
            rtree_idx = rtree_index.Index()
            rtree_idx.insert(0, merged.bounds)
            has_rtree = True
        except Exception:
            has_rtree = False

        mask = np.zeros((self.nlat, self.nlon), dtype=bool)
        minx, miny, maxx, maxy = merged.bounds
        inside_bbox = (
            (self.lats[:, None] >= miny) & (self.lats[:, None] <= maxy) &
            (self.lons[None, :] >= minx) & (self.lons[None, :] <= maxx))
        for i, lat in enumerate(self.lats):
            if not inside_bbox[i, :].any():
                continue
            for j, lon in enumerate(self.lons):
                if not inside_bbox[i, j]:
                    continue
                pt = Point(lon, lat)
                # Fast path: rtree candidate test
                if has_rtree:
                    candidates = list(rtree_idx.intersection(pt.bounds))
                    if not candidates:
                        continue
                # Use intersects (broader than contains) so boundary cells pass
                try:
                    mask[i, j] = merged.intersects(pt)
                except Exception:
                    try:
                        mask[i, j] = merged.contains(pt)
                    except Exception:
                        continue

        n_inside = int(mask.sum())
        if n_inside == 0:
            # Retry with a small buffer in case grid and polygon barely miss
            merged_buf = unary_union(gdf.geometry.tolist()).buffer(
                max(self.dlat, self.dlon) * 0.75)
            for i, lat in enumerate(self.lats):
                for j, lon in enumerate(self.lons):
                    pt = Point(lon, lat)
                    try:
                        mask[i, j] = merged_buf.intersects(pt)
                    except Exception:
                        continue
            n_inside = int(mask.sum())
            if n_inside > 0:
                merged = merged_buf

        self.mask = mask
        self.values[:, ~mask] = mask_value
        return mask

    def apply_mask(self, mask, mask_value=np.nan):
        self.mask = np.asarray(mask, dtype=bool)
        self.values[:, ~self.mask] = mask_value
        return self.mask

    def build_land_mask(self, min_time_coverage=0.05):
        """Estimate a land/observation mask from data coverage.

        A grid cell is retained if at least ``min_time_coverage`` (default
        5%) of its time entries are valid (non-NaN). Cells with essentially
        no data at all -- typically ocean/sea areas in land-only products
        such as CHIRTS -- are excluded. Coverage is accumulated in bounded
        time-chunks so large grids never allocate a (time, lat, lon) bool or
        float copy. The computed coverage is cached on ``self.data_coverage``.
        """
        nlat, nlon = self.nlat, self.nlon
        chunk = max(1, int(2 * 1024 ** 3 / max(1, nlat * nlon * 4)))
        valid = np.zeros((nlat, nlon), dtype=np.int64)
        for t0 in range(0, self.ntime, chunk):
            t1 = min(self.ntime, t0 + chunk)
            valid += np.isfinite(self.values[t0:t1]).sum(axis=0)
        cov = (valid.astype(np.float64) / max(1, self.ntime))
        mask = cov >= float(min_time_coverage)
        self.data_coverage = cov
        self.land_mask = mask
        return mask

    @staticmethod
    def _nearest_idx(axis, query):
        """Vectorised nearest-index lookup of ``query`` into ascending ``axis``."""
        a = np.asarray(axis, dtype=np.float64)
        q = np.asarray(query, dtype=np.float64)
        idx = np.searchsorted(a, q)
        idx = np.clip(idx, 1, len(a) - 1)
        left = a[idx - 1]
        right = a[idx]
        # prefer the nearer side; ties go to the upper neighbour
        use_left = np.abs(q - left) < np.abs(q - right)
        idx = idx - use_left.astype(np.int64)
        return idx

    @staticmethod
    def load_land_ocean_mask(path, grid, land_threshold=0.0):
        """Resample a static land/ocean mask file onto the grid (True = land).

        Supports GeoTIFF (rasterio) and NetCDF (netCDF4). For a GeoTIFF the
        nearest pixel is sampled per grid centroid and cells whose value is
        ``> land_threshold`` count as land/kept. For a NetCDF file the lat/lon
        coordinates are used for vectorised nearest-neighbour resampling.
        Returns a boolean ``(nlat, nlon)`` array.
        """
        path_l = str(path).lower()
        if path_l.endswith(('.nc', '.nc4', '.netcdf')):
            if not HAS_NETCDF4:
                raise ImportError('netCDF4 required to read NetCDF land masks.')
            ds = netCDF4.Dataset(path, 'r')
            try:
                lat_name = None
                lon_name = None
                mask_name = None
                for name, v in ds.variables.items():
                    if v.ndim == 2:
                        lat_name = lat_name or ('lat' in name.lower() and name)
                        lon_name = lon_name or ('lon' in name.lower() and name)
                        mask_name = mask_name or name
                    if mask_name and lat_name and lon_name:
                        break
                if not mask_name:
                    raise ValueError('No 2-D mask variable found in NetCDF land mask file.')
                if not lat_name or not lon_name:
                    lat_name, lon_name = getattr(ds.variables[mask_name], 'dimensions')
                m_lats = None
                m_lons = None
                if lat_name in ds.variables and lon_name in ds.variables:
                    m_lats = np.asarray(ds.variables[lat_name][:], dtype=float)
                    m_lons = np.asarray(ds.variables[lon_name][:], dtype=float)
                if m_lats is None or m_lons is None:
                    raise ValueError('Could not find lat/lon coordinates in '
                                     'the NetCDF land mask file.')
                data = np.asarray(ds.variables[mask_name][:], dtype=np.float32)
            finally:
                ds.close()
            data = np.squeeze(data)
            land = data > np.float32(land_threshold)
            flip_lat = m_lats[0] > m_lats[-1]
            flip_lon = m_lons[0] > m_lons[-1]
            land = land[::-1, :] if flip_lat else land
            land = land[:, ::-1] if flip_lon else land
            lat_axis = m_lats[::-1] if flip_lat else m_lats
            lon_axis = m_lons[::-1] if flip_lon else m_lons
            ri = HeatwaveGrid._nearest_idx(lat_axis, grid.lats)
            ci = HeatwaveGrid._nearest_idx(lon_axis, grid.lons)
            return land[np.ix_(ri, ci)]
        else:
            if not HAS_RASTERIO:
                raise ImportError('rasterio required to read GeoTIFF land masks.')
            with rasterio.open(path) as src:
                band = src.read(1)
                from rasterio.transform import rowcol
                xs = np.tile(np.asarray(grid.lons, dtype=np.float64), grid.nlat)
                ys = np.repeat(np.asarray(grid.lats, dtype=np.float64), grid.nlon)
                rows, cols = rowcol(src.transform, xs, ys)
                valid = ((rows >= 0) & (rows < band.shape[0]) &
                         (cols >= 0) & (cols < band.shape[1]))
                out = np.zeros((grid.nlat * grid.nlon), dtype=bool)
                flat = np.asarray(band, dtype=np.float32)[rows[valid], cols[valid]] > \
                    np.float32(land_threshold)
                out[valid] = flat
                return out.reshape((grid.nlat, grid.nlon))


# ----------------------------------------------------------------------------
#  NetCDF time conversion helpers
# ----------------------------------------------------------------------------

def _netcdf_times_to_datetime(values, units, calendar='standard'):
    values = np.asarray(values)
    if np.ma.is_masked(values):
        values = np.ma.filled(values, fill_value=np.nan)
    values = np.asarray(values, dtype=float)
    try:
        decoded = netCDF4.num2date(values, units, calendar=calendar,
                                   only_use_cftime_datetimes=True)
        return pd.to_datetime([d.isoformat() for d in decoded])
    except Exception:
        try:
            return pd.to_datetime(netCDF4.num2date(values, units, calendar=calendar))
        except Exception:
            return pd.to_datetime(values, unit='D', origin='unix')


# ----------------------------------------------------------------------------
#  Spatial MHW field builder
# ----------------------------------------------------------------------------

def _points_from_grid(grid):
    """Return flat lists of active (lat, lon) points (non-NaN at any time)."""
    data2d = grid.values.reshape(grid.ntime, -1)
    # valid if at least one observed non-nan value across the time axis
    valid = ~np.all(np.isnan(data2d), axis=0)
    idx = np.where(valid)[0]
    pts = []
    for k in idx:
        i = k // grid.nlon
        j = k % grid.nlon
        pts.append((k, i, j, grid.lats[i], grid.lons[j]))
    return pts


def _intensity_time_series(grid, t_ord, mhws, clim):
    """Compute daily anomaly time series (temp - clim seas) at one point."""
    thresh = clim['thresh']
    seas = clim['seas']
    temp = grid.values
    # fill nan with seas
    temp_f = np.where(np.isnan(temp), seas, temp)
    diff = temp_f - seas
    exceed = np.zeros_like(temp, dtype=bool)
    exceed = temp_f > thresh
    return diff, exceed


def _serialize_dates(mhws):
    """Convert datetime.date arrays inside an MHW dict to strings."""
    out = dict(mhws)
    for key in ('date_start', 'date_end', 'date_peak'):
        if key in out:
            out[key] = [str(d) for d in out[key]]
    return out


def clean_points_kwargs(grid, clim_start, clim_end, pctile, min_duration,
                        join_across_gaps, max_gap, smooth_percentile,
                        smooth_percentile_width, window_half_width):
    return {
        'climatologyPeriod': [int(clim_start), int(clim_end)],
        'pctile': float(pctile),
        'minDuration': int(min_duration),
        'joinAcrossGaps': bool(join_across_gaps),
        'smoothPercentile': bool(smooth_percentile),
        'smoothPercentileWidth': int(smooth_percentile_width),
        'windowHalfWidth': int(window_half_width),
    } | ({'maxGap': int(max_gap)} if join_across_gaps else {})


class SpatialHeatwaveResult:
    """Container for spatially-explicit heatwave metrics & maps."""

    def __init__(self, grid, mask, mhws_map=None, metric_maps=None,
                 params=None, error_map=None):
        self.grid = grid
        self.mask = mask
        self.mhws_map = mhws_map or {}
        self.metric_maps = metric_maps or {}
        self.params = params or {}
        self.error_map = error_map if error_map is not None else \
            np.zeros((grid.nlat, grid.nlon), dtype=bool)

    # -------- Metric map helpers --------
    def _metric_array(self, key):
        arr = np.full((self.grid.nlat, self.grid.nlon), np.nan)
        for (i, j), val in self.metric_maps.get(key, {}).items():
            arr[i, j] = val
        return arr

    def metric_maps_as_dict(self, keys=None):
        """Flatten metric maps into {lon, lat, value} JSON-friendly arrays."""
        keys = keys or list(self.metric_maps.keys())
        out = {}
        for key in keys:
            arr = self._metric_array(key)
            out[key] = {
                'lats': self.grid.lats.tolist(),
                'lons': self.grid.lons.tolist(),
                'values': arr.tolist()
            }
        return out

    # -------- Automated summarize --------
    def summarize(self, num_points=None):
        """Return region summary statistics for display."""
        summary = {}
        for key, cells in self.metric_maps.items():
            vals = np.array(list(cells.values()))
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            summary[key] = {
                'min': float(np.nanmin(vals)),
                'max': float(np.nanmax(vals)),
                'mean': float(np.nanmean(vals)),
                'median': float(np.nanmedian(vals)),
                'std': float(np.nanstd(vals)),
                'total': float(np.nansum(vals)),
            }
        return summary

    def event_table(self, limit=None, sort_by='duration'):
        """Human readable event table (top events sorted by metric)."""
        rows = []
        for (i, j), d in self.metric_maps.get('n_events', {}).items():
            if not d or d is None or np.isnan(d):
                continue
            rows.append({
                'lat': float(self.grid.lats[i]),
                'lon': float(self.grid.lons[j]),
                'total_events': int(d),
                'avg_duration_days': round(float(self.metric_maps['mean_duration'][(i, j)]), 2)
                    if 'mean_duration' in self.metric_maps else None,
                'avg_max_intensity': round(float(self.metric_maps['mean_intensity_max'][(i, j)]), 3)
                    if 'mean_intensity_max' in self.metric_maps else None,
            })
        if sort_by and rows:
            rows.sort(key=lambda r: r.get(sort_by) or 0, reverse=True)
        return rows[:limit] if limit else rows


# ----------------------------------------------------------------------------
#  Core spatial detection runner
# ----------------------------------------------------------------------------

_POOL_CTX = {}


def _detect_pool_init(grid, t_ord, kwargs):
    """Pool-worker initializer: shares the grid without per-task pickling."""
    _POOL_CTX['grid'] = grid
    _POOL_CTX['t_ord'] = t_ord
    _POOL_CTX['kwargs'] = kwargs
    _POOL_CTX['prepared'] = True


def _detect_pool_worker(ij):
    """Detect heatwaves for a single (i, j) cell inside a pool worker."""
    i, j = int(ij[0]), int(ij[1])
    grid = _POOL_CTX['grid']
    t_ord = _POOL_CTX['t_ord']
    kwargs = _POOL_CTX['kwargs']
    series = np.asarray(grid.values[:, i, j], dtype=float)
    if np.isnan(series).all() or np.std(series[~np.isnan(series)]) == 0:
        return (i, j, None, None, 'no-valid-data')
    try:
        mh, cl = mhw.detect(t_ord, series.copy(), **kwargs)
        return (i, j, mh, cl, None)
    except Exception as exc:
        return (i, j, None, None, str(exc)[:200])


def _default_cancel_cb():
    return False


def run_spatial_detection(grid, clim_start, clim_end, pctile=90,
                          min_duration=3, join_across_gaps=True, max_gap=2,
                          smooth_percentile=True, smooth_percentile_width=31,
                          window_half_width=5, mask=None, progress_cb=None,
                          max_points=None, n_jobs=1, cancel_cb=None,
                          use_process_pool=None, process_memory_limit_gb=2.0):
    """Run MHW detection over every grid point (optionally in parallel).

    Returns a dict of per-point MHW results keyed by (i, j) -> (mhws, clim)
    plus aggregated maps. ``n_jobs`` cores, a ``progress_cb(done, total)`` and
    a ``cancel_cb() -> bool`` are honoured. When ``cancel_cb()`` returns True
    the run stops early and the returned result is marked ``cancelled=True``.
    Large grids are processed with a thread pool (keeps a single shared copy
    of the grid); grids whose float32 working set fits below
    ``process_memory_limit_gb`` use a process pool for CPU parallelism.
    """
    cancel_cb = cancel_cb or _default_cancel_cb
    if mask is None:
        mask = np.ones((grid.nlat, grid.nlon), dtype=bool)

    t_ord = grid.t_ordinal()

    active_rows, active_cols = np.where(mask)
    total = len(active_rows)
    flat_results = {}
    n_ok = 0
    n_failed = 0
    error_map = np.zeros((grid.nlat, grid.nlon), dtype=bool)
    cancelled = False

    kwargs = clean_points_kwargs(
        grid, clim_start, clim_end, pctile, min_duration,
        join_across_gaps, max_gap, smooth_percentile,
        smooth_percentile_width, window_half_width)

    def accumulate(i, j, mh, cl, err):
        nonlocal n_ok, n_failed
        if err is not None or mh is None:
            n_failed += 1
            error_map[i, j] = True
        else:
            flat_results[(i, j)] = (mh, cl)
            n_ok += 1

    try:
        if n_jobs <= 1 or total <= 1:
            for k in range(total):
                if cancel_cb():
                    cancelled = True
                    break
                if progress_cb is not None and (k % max(1, total // 20) == 0):
                    progress_cb(k, total)
                i = int(active_rows[k])
                j = int(active_cols[k])
                series = np.asarray(grid.values[:, i, j], dtype=float)
                if np.isnan(series).all() or np.std(series[~np.isnan(series)]) == 0:
                    accumulate(i, j, None, None, 'no-valid-data')
                    continue
                try:
                    mh, cl = mhw.detect(t_ord, series.copy(), **kwargs)
                    accumulate(i, j, mh, cl, None)
                except Exception as exc:
                    accumulate(i, j, None, None, str(exc)[:200])
            if progress_cb is not None and not cancelled:
                progress_cb(total, total)
        else:
            est_bytes = grid.values.nbytes
            use_proc = (use_process_pool if use_process_pool is not None
                        else est_bytes <= process_memory_limit_gb * 1024 ** 3)
            Executor = ProcessPoolExecutor if use_proc else ThreadPoolExecutor
            from concurrent.futures import as_completed
            initargs = (grid, t_ord, kwargs)
            pooled = False
            try:
                with Executor(max_workers=int(n_jobs),
                              initializer=_detect_pool_init,
                              initargs=initargs) as ex:
                    fut_to_ij = {
                        ex.submit(_detect_pool_worker,
                                  (int(active_rows[k]), int(active_cols[k]))):
                        (int(active_rows[k]), int(active_cols[k]))
                        for k in range(total)
                    }
                    done = 0
                    for fut in as_completed(fut_to_ij):
                        if cancel_cb():
                            cancelled = True
                            ex.shutdown(wait=False, cancel_futures=True)
                            break
                        done += 1
                        if progress_cb is not None and (done % max(1, total // 20) == 0):
                            progress_cb(done, total)
                        i, j = fut_to_ij[fut]
                        try:
                            ri, rj, mh, cl, err = fut.result()
                            accumulate(ri, rj, mh, cl, err)
                        except Exception as exc:
                            accumulate(i, j, None, None, str(exc)[:200])
                pooled = True
            except Exception:
                # Pool setup failed (e.g. spawn pickling) -> sequential fallback
                pooled = False
            if not pooled and not cancelled:
                for k in range(total):
                    if cancel_cb():
                        cancelled = True
                        break
                    i = int(active_rows[k])
                    j = int(active_cols[k])
                    series = np.asarray(grid.values[:, i, j], dtype=float)
                    if np.isnan(series).all() or np.std(series[~np.isnan(series)]) == 0:
                        accumulate(i, j, None, None, 'no-valid-data')
                        continue
                    try:
                        mh, cl = mhw.detect(t_ord, series.copy(), **kwargs)
                        accumulate(i, j, mh, cl, None)
                    except Exception as exc:
                        accumulate(i, j, None, None, str(exc)[:200])
                if progress_cb is not None and not cancelled:
                    progress_cb(total, total)
    except Exception:
        flat_results = {}
        n_ok = 0
        n_failed = 0
        error_map = np.zeros((grid.nlat, grid.nlon), dtype=bool)
        cancelled = True
        raise

    # -------- Compute metric maps --------
    metric_maps = {key: {} for key in (
        'n_events', 'mean_duration', 'max_duration',
        'mean_intensity_max', 'max_intensity_max', 'mean_intensity_mean',
        'cumulative_intensity', 'total_hw_days',
        'mean_year_freq', 'max_intensity_abs', 'categories_count')}

    for (i, j), (mh, cl) in flat_results.items():
        if mh.get('n_events', 0) == 0:
            metric_maps['n_events'][(i, j)] = 0
            continue
        dur = np.array(mh['duration'])
        imax = np.array(mh['intensity_max'])
        imean = np.array(mh['intensity_mean'])
        icum = np.array(mh['intensity_cumulative'])
        iabs = np.array(mh['intensity_max_abs'])
        start_dates = [d.year for d in mh['date_start']]
        n_events = len(dur)
        years_span = (max(start_dates) - min(start_dates) + 1) if start_dates else 1

        metric_maps['n_events'][(i, j)] = float(n_events)
        metric_maps['mean_duration'][(i, j)] = float(np.mean(dur))
        metric_maps['max_duration'][(i, j)] = float(np.max(dur))
        metric_maps['mean_intensity_max'][(i, j)] = float(np.mean(imax))
        metric_maps['max_intensity_max'][(i, j)] = float(np.max(imax))
        metric_maps['mean_intensity_mean'][(i, j)] = float(np.mean(imean))
        metric_maps['cumulative_intensity'][(i, j)] = float(np.sum(icum))
        metric_maps['total_hw_days'][(i, j)] = float(np.sum(dur))
        metric_maps['mean_year_freq'][(i, j)] = float(n_events / years_span)
        metric_maps['max_intensity_abs'][(i, j)] = float(np.max(iabs))
        # categories (moderate=1, strong=2, severe=3, extreme=4)
        cats = [1 if c == 'Moderate' else 2 if c == 'Strong' else 3 if c == 'Severe' else 4
                for c in mh.get('category', [])]
        metric_maps['categories_count'][(i, j)] = float(max(cats)) if cats else 0

    result = SpatialHeatwaveResult(
        grid, mask, mhws_map=flat_results, metric_maps=metric_maps,
        params=dict(clim_start=clim_start, clim_end=clim_end, pctile=pctile,
                    min_duration=min_duration, join_across_gaps=join_across_gaps,
                    max_gap=max_gap), error_map=error_map)
    result._counts = (n_ok, n_failed)
    result.cancelled = cancelled
    return result


# ----------------------------------------------------------------------------
#  Annual / seasonal statistics builder
# ----------------------------------------------------------------------------

def build_annual_field_stats(grid, result, year_start=None, year_end=None,
                             block_length=1):
    """Build per-year (or per-block) grid maps of {count, days, cumint, intensity}."""
    years = np.unique(grid.times.year)
    years = [y for y in years if (year_start is None or y >= year_start)
             and (year_end is None or y <= year_end)]
    if not years:
        years = [int(grid.times.year.min()), int(grid.times.year.max())]
    blocks = []
    for start in range(int(years[0]), int(years[-1]) + 1, block_length):
        blocks.append((start, start + block_length - 1))
    if not blocks:
        blocks = [(int(years[0]), int(years[0]))]

    # Allocate map containers
    count_map = {}
    days_map = {}
    cumint_map = {}
    imax_map = {}
    freq_map = {}

    for (i, j), (mh, cl) in result.mhws_map.items():
        starts = np.array([d.year for d in mh['date_start']])
        durs = np.array(mh['duration'])
        icums = np.array(mh['intensity_cumulative'])
        imaxs = np.array(mh['intensity_max'])
        n_events = mh.get('n_events', 0)
        for (b0, b1) in blocks:
            sel = (starts >= b0) & (starts <= b1) if n_events else np.zeros(0, dtype=bool)
            count_map[(i, j, b0)] = float(sel.sum())
            days_map[(i, j, b0)] = float(durs[sel].sum()) if sel.any() else 0.0
            cumint_map[(i, j, b0)] = float(icums[sel].sum()) if sel.any() else 0.0
            imax_map[(i, j, b0)] = float(imaxs[sel].max()) if sel.any() else 0.0
            span_years = max(1, sum(1 for y in years if b0 <= y <= b1))
            freq_map[(i, j, b0)] = float(sel.sum() / span_years)

    return {
        'blocks': blocks,
        'count': count_map,
        'days': days_map,
        'cumint': cumint_map,
        'imax': imax_map,
        'freq': freq_map,
    }


def grids_from_year_stats(stat_grid, grid, key, blocks):
    """Convert a {(i,j,b0): val} map into a 2D array per block."""
    out = []
    for (b0, b1) in blocks:
        arr = np.full((grid.nlat, grid.nlon), np.nan)
        for (i, j, b), v in stat_grid[key].items():
            if b == b0:
                arr[i, j] = v
        out.append({'start': b0, 'end': b1, 'values': arr})
    return out


# ----------------------------------------------------------------------------
#  Spatial trends
# ----------------------------------------------------------------------------

def _single_trend(x, y, alpha=0.05):
    """Return (slope, intercept, r_value, p_value, std_err) or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return None
    xx, yy = x[valid], y[valid]
    try:
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(xx, yy)
        return (float(slope), float(intercept), float(r_value),
                float(p_value), float(std_err))
    except Exception:
        return None


def compute_spatial_trends(grid, result, metric='n_events',
                           clim_start=None, clim_end=None,
                           use_years=None, alpha=0.05):
    """Compute per-cell linear trends of a given metric over time.

    metric keys available:
        'n_events'   -> events per year
        'total_hw_days' -> days per year
        'cumulative_intensity' -> sum intensity per year
        'mean_intensity_max' -> mean of peak intensity per year
    For per-year values we build a year grid from the metric maps plus
    a year lookup built from event start dates.
    """
    if metric in ('n_events', 'total_hw_days', 'cumulative_intensity',
                  'mean_intensity_max', 'mean_intensity_mean'):
        return _trends_from_events(grid, result, metric, clim_start,
                                   clim_end, use_years, alpha)

    raise ValueError(f'Unsupported trend metric: {metric}')


def _event_year_series(result, cell, metric, years_list):
    """Produce (years, values) arrays for a given cell & metric."""
    mh = result.mhws_map[cell][0]
    n_events = mh.get('n_events', 0)
    if n_events == 0:
        return None
    starts = np.array([d.year for d in mh['date_start']])
    durs = np.array(mh['duration'])
    imaxs = np.array(mh['intensity_max'])
    imeans = np.array(mh['intensity_mean'])
    icums = np.array(mh['intensity_cumulative'])

    vals = {y: 0.0 for y in years_list}
    for t0, d0 in zip(starts, durs):  # not used directly
        pass

    if metric == 'n_events':
        for y in years_list:
            vals[y] = float(np.sum(starts == y))
    elif metric == 'total_hw_days':
        for y in years_list:
            vals[y] = float(np.sum(durs[starts == y]))
    elif metric == 'cumulative_intensity':
        for y in years_list:
            vals[y] = float(np.sum(icums[starts == y]))
    elif metric == 'mean_intensity_max':
        for y in years_list:
            sel = starts == y
            vals[y] = float(np.mean(imaxs[sel])) if sel.any() else 0.0
    elif metric == 'mean_intensity_mean':
        for y in years_list:
            sel = starts == y
            vals[y] = float(np.mean(imeans[sel])) if sel.any() else 0.0
    return years_list, [vals[y] for y in years_list]


def _trends_from_events(grid, result, metric, clim_start, clim_end,
                        use_years, alpha):
    years_all = np.unique(grid.times.year)
    years = list(years_all)
    if clim_start is not None:
        years = [y for y in years if y >= int(clim_start)]
    if clim_end is not None:
        years = [y for y in years if y <= int(clim_end)]
    if use_years is not None and len(use_years) >= 3:
        years = [int(y) for y in use_years]

    trend_map = {}
    pvalue_map = {}
    slope_map = {}
    r2_map = {}

    for (i, j) in result.mhws_map.keys():
        series = _event_year_series(result, (i, j), metric, years)
        if series is None:
            continue
        yv, mvals = series
        res = _single_trend(yv, mvals, alpha)
        if res is None:
            continue
        slope, intercept, r_value, p_value, std_err = res
        trend_map[(i, j)] = float(slope)       # per year
        pvalue_map[(i, j)] = float(p_value)
        slope_map[(i, j)] = float(slope)
        r2_map[(i, j)] = float(r_value ** 2)

    return {
        'metric': metric,
        'years': years,
        'test': 'mann-kendall' if False else 'linear-regression',
        'slope': slope_map,
        'pvalue': pvalue_map,
        'r2': r2_map,
        'trend': trend_map,
    }


def mann_kendall(x, y, alpha=0.05):
    """Mann-Kendall trend test; returns (slope, p, trend_dir) or None."""
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 4:
            return None
        res = sp_stats.kendalltau(x[valid], y[valid])
        trend = 'increasing' if res.statistic > 0 else 'decreasing'
        return (float(res.statistic), float(res.pvalue), trend)
    except Exception:
        return None


def mann_kendall_field(grid, result, metric='n_events', years=None,
                       clim_start=None, clim_end=None, alpha=0.05):
    """Spatial field of Mann-Kendall statistics for a metric."""
    years_all = list(np.unique(grid.times.year))
    if clim_start is not None:
        years_all = [y for y in years_all if y >= int(clim_start)]
    if clim_end is not None:
        years_all = [y for y in years_all if y <= int(clim_end)]
    if years is not None and len(years) >= 4:
        years_all = [int(y) for y in years]

    tau_map = {}
    p_map = {}
    direction_map = {}
    for (i, j) in result.mhws_map.keys():
        series = _event_year_series(result, (i, j), metric, years_all)
        if series is None:
            continue
        yv, mvals = series
        mk = mann_kendall(yv, mvals, alpha)
        if mk is None:
            continue
        tau, p, direction = mk
        tau_map[(i, j)] = tau
        p_map[(i, j)] = p
        direction_map[(i, j)] = direction
    return {'metric': metric, 'years': years_all,
            'tau': tau_map, 'pvalue': p_map, 'direction': direction_map}


# ----------------------------------------------------------------------------
#  Comparative analysis between two fields
# ----------------------------------------------------------------------------

def grid_stats_table(result_a, result_b=None, metric_keys=None):
    """Build a per-metric table of A / B statistics for comparative analysis."""
    metric_keys = metric_keys or [
        'n_events', 'total_hw_days', 'mean_duration', 'mean_intensity_max',
        'cumulative_intensity', 'mean_year_freq']
    if result_b is None:
        # single-field stats
        return result_a.summarize()

    summary_a = result_a.summarize()
    summary_b = result_b.summarize()

    rows = []
    for key in metric_keys:
        row = {
            'metric': key,
            'a_mean': None, 'a_max': None, 'a_total': None,
            'b_mean': None, 'b_max': None, 'b_total': None,
            'diff_mean': None, 'ratio_mean': None,
        }
        sa = summary_a.get(key)
        sb = summary_b.get(key)
        if sa:
            row.update({'a_mean': sa['mean'], 'a_max': sa['max'],
                        'a_total': sa['total']})
        if sb:
            row.update({'b_mean': sb['mean'], 'b_max': sb['max'],
                        'b_total': sb['total']})
        if sa and sb:
            row['diff_mean'] = float(sa['mean'] - sb['mean'])
            if sb['mean']:
                row['ratio_mean'] = float(sa['mean'] / sb['mean'])
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
#  JSON-safe serialization helpers
# ----------------------------------------------------------------------------

def nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def float_round_list(arr, ndigits=4):
    return [round(float(v), ndigits) if not np.isnan(v) else None
            for v in arr]
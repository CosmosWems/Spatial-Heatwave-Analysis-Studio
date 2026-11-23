"""
Spatial Heatwave Web Application
================================
A Flask web application for the spatially-explicit detection, statistical
analysis, trend assessment and visualisation of atmospheric heatwaves over a
user-defined study region (masked by an administrative-boundary shapefile).

The application is built on the marineHeatWaves (Hobday et al., 2016)
methodology but operates on a gridded (lat, lon, time) temperature field and
computes a heatwave statistic for *every* grid point so that the spatial
dynamics of heatwaves over the region can be mapped and animated.

Sections
--------
  * Data Upload & Preprocessing   (NetCDF / GeoTIFF stack + shapefile)
  * Heatwave Computations         (Detect / Statistics / Trends buttons)
  * Visualization                 (metric maps, point series, animations)
  * Comparative Analysis          (period comparisons, difference maps)
"""

import io
import json
import os
import time
import threading
import multiprocessing
import tempfile
import zipfile
import datetime
import base64
import struct as _struct
import math as _math
import traceback
import warnings

import numpy as np
import pandas as pd

import plotly.graph_objs as go
import plotly.express as px

from flask import (Flask, request, jsonify, render_template_string,
                   send_file, render_template)

import marineHeatWaves as mhw

import spatial_heatwave_engine as she

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = "spatial_hw_secret_key"


# ----------------------------------------------------------------------------
#  Global state
# ----------------------------------------------------------------------------

APP_STATE = {
    'grid': None,                 # HeatwaveGrid, primary (reference) field
    'grid_info': None,            # dict of metadata shown in UI
    'mask': None,                 # numpy bool (nlat, nlon) - study area mask
    'preprocess_report': None,    # dict
    'spatial_result': None,       # SpatialHeatwaveResult from detection
    'trends': None,               # dict of trend fields
    'year_stats': None,           # dict of annual/block stat maps
    'year_frames': None,          # cached animation frames
    'secondary_grid': None,       # second dataset for comparative analysis
    'secondary_info': None,
    'secondary_result': None,
    'tmp_shapefile_dir': None,
    'tmp_geo': {},                # preview geographies (admin), cache by name
    'nc_path': None,              # path to uploaded NetCDF for re-loading
    'shp_path': None,             # path to uploaded shapefile
    'shp_gdf_summary': None,
    'land_mask_path': None,       # path to uploaded static land/ocean mask
    'land_mask_static': None,     # boolean (nlat, nlon) True=land
}


# Detect-job state shared with the frontend polling loop.
_COMPUTE_LOCK = threading.Lock()
COMPUTE_STATE = {
    'running': False,             # a detect job thread is alive
    'phase': 'idle',              # idle|starting|detecting|done|error|cancelled
    'done': 0,
    'total': 0,
    'percent': 0.0,
    'label': 'idle',
    'elapsed_ms': 0,
    'started_at': None,
    'finished_at': None,
    'cancel_requested': False,
    'cancelled': False,
    'error': None,
    'n_jobs': 1,
    'masked_cells': 0,
    'last_summary': None,
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(_APP_DIR, '_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload(file_storage, folder_name):
    """Persist an uploaded file into a temp folder and return its path."""
    folder = os.path.join(UPLOAD_DIR, folder_name, str(os.getpid()))
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_storage.filename)
    file_storage.save(path)
    return path


def _find_archive_member(root, suffixes):
    """Find a supported file anywhere below an extracted upload directory."""
    matches = []
    for current_root, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(suffixes):
                matches.append(os.path.join(current_root, name))
    return sorted(matches, key=lambda path: (os.path.basename(path).lower(), path.lower()))


def _read_boundary_file(path):
    """Read a boundary upload from a ZIP, shapefile, or GeoJSON file."""
    if not she.HAS_GEOPANDAS:
        raise ImportError('geopandas not installed. pip install geopandas')
    gpd = __import__('geopandas')
    # Allow a standalone .shp upload when GDAL can rebuild its missing .shx.
    os.environ.setdefault('SHAPE_RESTORE_SHX', 'YES')
    lower_path = path.lower()
    if lower_path.endswith('.zip'):
        extract_dir = os.path.join(os.path.dirname(path), 'shp_extract')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(path, 'r') as zf:
            members = [m for m in zf.namelist() if not m.endswith('/')]
            if not any(m.lower().endswith('.shp') for m in members):
                raise ValueError('No .shp file found inside the ZIP archive. Include the .shp, .shx, and .dbf files.')
            root_real = os.path.realpath(extract_dir) + os.sep
            for member in members:
                target = os.path.realpath(os.path.join(extract_dir, member))
                if not target.startswith(root_real):
                    raise ValueError('ZIP archive contains an unsafe path.')
            zf.extractall(extract_dir)
        shp_files = _find_archive_member(extract_dir, ('.shp',))
        if not shp_files:
            raise ValueError('No .shp file found inside the ZIP archive.')
        return gpd.read_file(shp_files[0]), extract_dir
    if lower_path.endswith(('.geojson', '.gjson', '.json')):
        return gpd.read_file(path), os.path.dirname(path)
    if lower_path.endswith('.shp'):
        return gpd.read_file(path), os.path.dirname(path)
    raise ValueError('Unsupported boundary format. Upload .zip, .shp, .geojson, or .gjson.')


# ----------------------------------------------------------------------------
#  Small helpers
# ----------------------------------------------------------------------------

def _grid_json_info(grid):
    nt, nlat, nlon = grid.shape
    return {
        'variable': grid.variable,
        'ntime': int(nt),
        'nlat': int(nlat),
        'nlon': int(nlon),
        'lat_min': float(np.nanmin(grid.lats)),
        'lat_max': float(np.nanmax(grid.lats)),
        'lon_min': float(np.nanmin(grid.lons)),
        'lon_max': float(np.nanmax(grid.lons)),
        't_start': str(grid.times.min()),
        't_end': str(grid.times.max()),
        'n_years': int(len(np.unique(grid.times.year))),
        'mean': float(np.nanmean(grid.values)) if nt else 0.0,
        'nan_frac': float(np.isnan(grid.values).mean()) if nt else 1.0,
        'dlon': grid.dlon,
        'dlat': grid.dlat,
        'source_units': getattr(grid, 'source_units', ''),
        'converted_from_kelvin': bool(getattr(grid, 'converted_from_kelvin', False)),
    }


def _require_grid():
    grid = APP_STATE.get('grid')
    if grid is None:
        raise RuntimeError('No temperature grid loaded. Upload NetCDF or a '
                           'GeoTIFF stack first (Section 01).')
    return grid


def _require_result():
    res = APP_STATE.get('spatial_result')
    if res is None:
        raise RuntimeError('No spatial detection results yet. Run the '
                           '"Detect Heatwaves" computation first.')
    return res


def _requested_cores(n_jobs):
    """Resolve n_jobs (0/None => all cores, otherwise clamp to available)."""
    try:
        n = int(n_jobs)
    except (TypeError, ValueError):
        n = 0
    available = max(1, multiprocessing.cpu_count())
    if n <= 0:
        n = available
    return max(1, min(n, available))


# ----------------------------------------------------------------------------
#  Background detection job
# ----------------------------------------------------------------------------

def _detect_worker(params):
    """Run detection off the request thread, updating COMPUTE_STATE."""
    start = time.time()
    try:
        grid = params['grid']
        mask = params['mask']
        n_jobs = params['n_jobs']

        def progress_cb(done, total):
            with _COMPUTE_LOCK:
                COMPUTE_STATE['done'] = int(done)
                COMPUTE_STATE['total'] = int(total)
                COMPUTE_STATE['percent'] = round(done / total * 100, 2) if total else 0.0
                COMPUTE_STATE['elapsed_ms'] = int((time.time() - start) * 1000)
                COMPUTE_STATE['label'] = f'Detecting ... {done} / {total} cells ({COMPUTE_STATE["percent"]:.1f}%)'
                COMPUTE_STATE['phase'] = 'detecting'
                COMPUTE_STATE['n_jobs'] = n_jobs

        def cancel_cb():
            with _COMPUTE_LOCK:
                return bool(COMPUTE_STATE.get('cancel_requested'))

        result = she.run_spatial_detection(
            grid, clim_start=params['clim_start'], clim_end=params['clim_end'],
            pctile=params['pctile'], min_duration=params['min_duration'],
            join_across_gaps=params['join_across_gaps'],
            max_gap=params['max_gap'],
            smooth_percentile=params['smooth_percentile'],
            smooth_percentile_width=params['smooth_pctile_width'],
            window_half_width=params['window_half_width'],
            mask=mask, progress_cb=progress_cb, cancel_cb=cancel_cb,
            n_jobs=n_jobs)

        if getattr(result, 'cancelled', False):
            with _COMPUTE_LOCK:
                COMPUTE_STATE.update(
                    running=False, phase='cancelled', cancelled=True,
                    finished_at=time.time(),
                    elapsed_ms=int((time.time() - start) * 1000),
                    label='Detection cancelled by user.')
            return

        APP_STATE['spatial_result'] = result
        ok, failed = result._counts
        summary = result.summarize()
        with _COMPUTE_LOCK:
            COMPUTE_STATE.update(
                running=False, phase='done', cancelled=False,
                error=None, percent=100.0,
                done=ok, total=ok + failed,
                finished_at=time.time(),
                elapsed_ms=int((time.time() - start) * 1000),
                label=f'Detection complete: {ok} cells OK, {failed} failed.',
                last_summary={
                    'event_table': result.event_table(limit=50),
                    'summary': summary,
                    'processed_cells': int(ok),
                    'error_cells': int(failed),
                    'metric_keys': list(result.metric_maps.keys()),
                    'params': result.params,
                    'total_events_region': float(np.nansum(
                        list(result.metric_maps['n_events'].values())) or 0),
                })
    except Exception as exc:
        traceback.print_exc()
        with _COMPUTE_LOCK:
            COMPUTE_STATE.update(
                running=False, phase='error', cancelled=False,
                error=str(exc), finished_at=time.time(),
                elapsed_ms=int((time.time() - start) * 1000),
                label=f'Detection error: {exc}')


@app.errorhandler(RuntimeError)
def _runtime_error_handler(exc):
    """Return RuntimeErrors (missing state, etc.) as JSON, not HTML."""
    if request.path.startswith('/api/'):
        return jsonify({'error': str(exc)}), 400
    return jsonify({'error': str(exc)}), 400


def _apply_study_area_mask(grid, buffer_degrees=0.0):
    """Extract the study area from a grid using the uploaded shapefile.

    Cells outside the boundary become NaN; the boolean mask is stored in
    APP_STATE['mask'] so detection only runs inside the study area.
    Returns (grid, mask); the grid is the same object after masking.
    """
    if grid is None:
        return grid, None
    shp = APP_STATE.get('shp_path')
    if not shp:
        return grid, None
    mask = grid.mask_with_shapefile(shp, buffer_degrees=buffer_degrees)
    APP_STATE['mask'] = mask
    return grid, mask


def _field_figure(values, lats, lons, title, colorscale='RdYlBu_r',
                  label='value', zmin=None, zmax=None, show_geo=True,
                  cmin=None, cmax=None, unit='', center=None):
    """Make a filled-contour/heatmap figure of a 2D field over lat/lon."""
    fig = go.Figure()
    trace = go.Heatmap(
        x=lons, y=lats, z=values,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=dict(text=unit, side='right'),
                      thickness=14, len=0.7, tickfont=dict(size=10)),
        hovertemplate='Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>%{z:.3f}' + unit + '<extra></extra>'
    )
    fig.add_trace(trace)

    if show_geo:
        if 'boundaries' in APP_STATE.get('geo_figure_data', {}):
            geo = APP_STATE['geo_figure_data']['boundaries']
            for ring in geo:
                fig.add_trace(go.Scatter(
                    x=ring.get('lons', []), y=ring.get('lats', []),
                    mode='lines', line=dict(color='#2f3b3a', width=2),
                    hoverinfo='skip', showlegend=False,
                    fill='none'
                ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0c1714', plot_bgcolor='#070e0c',
        title=dict(text=f'<b>{title}</b>', font=dict(size=15)),
        xaxis_title='Longitude (\u00b0E)',
        yaxis_title='Latitude (\u00b0N)',
        margin=dict(l=60, r=70, t=55, b=50),
        height=460,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor='x'),
    )
    return fig


# ----------------------------------------------------------------------------
#  Routes: landing & home
# ----------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('spatial_index.html')


@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'modules': {
            'netCDF4': she.HAS_NETCDF4,
            'rasterio': she.HAS_RASTERIO,
            'geopandas': she.HAS_GEOPANDAS,
            'xarray': she.HAS_XARRAY,
            'scipy': she.HAS_SCIPY,
        }
    })


# ----------------------------------------------------------------------------
#  01. DATA UPLOAD & PREPROCESSING
# ----------------------------------------------------------------------------

@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Upload the temperature dataset (NetCDF or GeoTIFF stack)."""
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No data file provided.'}), 400

        filename = file.filename.lower()
        var_name = request.form.get('variable') or None
        temperature_unit = request.form.get('temperature_unit', 'auto')
        if filename.endswith('.nc'):
            path = _save_upload(file, 'data')
            grid = she.HeatwaveGrid.from_netcdf(
                path, var_name=var_name, temperature_unit=temperature_unit)
            source = 'netcdf'
            APP_STATE['nc_path'] = path
        elif filename.endswith('.tif') or filename.endswith('.tiff'):
            path = _save_upload(file, 'data')
            grid = she.HeatwaveGrid.from_geotiff_stack([path])
            source = 'geotiff'
            APP_STATE['nc_path'] = None
        elif filename.endswith('.zip'):
            path = _save_upload(file, 'data')
            with zipfile.ZipFile(path, 'r') as zf:
                names = [n for n in zf.namelist()
                         if n.lower().endswith(('.tif', '.tiff'))]
                if not names:
                    raise ValueError('ZIP archive must contain .tif/.tiff files.')
                extract_dir = os.path.join(os.path.dirname(path), 'tiffs')
                os.makedirs(extract_dir, exist_ok=True)
                zf.extractall(extract_dir)
                tiff_paths = [os.path.join(extract_dir, n) for n in names]
            grid = she.HeatwaveGrid.from_geotiff_stack(tiff_paths)
            source = 'geotiff_zip'
            APP_STATE['nc_path'] = None
        else:
            return jsonify({
                'error': 'Unsupported format. Upload a NetCDF (.nc), a single '
                         'GeoTIFF (.tif), or a ZIP archive of GeoTIFFs.'}), 400

        if grid.ntime < 60:
            raise ValueError('Dataset has fewer than 60 time steps; a longer '
                             'daily time series is needed for heatwave '
                             'climatology.')

        grid.add_noise_if_needed()

        # Extract the study area as soon as both inputs are available;
        # a previously uploaded shapefile is re-applied to the new data.
        if APP_STATE.get('shp_path'):
            _apply_study_area_mask(grid)

        APP_STATE['grid'] = grid
        APP_STATE['spatial_result'] = None
        APP_STATE['trends'] = None
        APP_STATE['year_stats'] = None

        info = _grid_json_info(grid)
        status = {
            'ok': True,
            'source': source,
            'info': info,
            'columns': ['time', 'lat', 'lon', 'value'],
            'missing_statistics': [
                {'parameter': grid.variable,
                 'missing': int(np.isnan(grid.values).sum()),
                 'available': int(np.isfinite(grid.values).sum()),
                 'total': int(grid.values.size),
                 'missing_percent': round(float(np.isnan(grid.values).mean()) * 100, 2)}
            ],
            'preview_table': _grid_preview_table(grid),
            'selected_variable': grid.variable,
            'temperature_unit': temperature_unit,
        }
        return jsonify(status)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _grid_preview_table(grid, rows=8):
    """Sample of the lon-lat grid for a preview table."""
    nlat, nlon = grid.nlat, grid.nlon
    ri = np.linspace(0, max(0, nlat - 1), min(rows, nlat)).astype(int)
    rj = np.linspace(0, max(0, nlon - 1), min(rows, nlon)).astype(int)
    data = []
    mid_t = grid.ntime // 2
    for i in ri:
        for j in rj:
            data.append({
                'lat': round(float(grid.lats[i]), 4),
                'lon': round(float(grid.lons[j]), 4),
                't0_value': round(float(grid.values[0, i, j]), 3)
                        if not np.isnan(grid.values[0, i, j]) else None,
                'tmid_value': round(float(grid.values[mid_t, i, j]), 3)
                        if not np.isnan(grid.values[mid_t, i, j]) else None,
                'tlast_value': round(float(grid.values[-1, i, j]), 3)
                        if not np.isnan(grid.values[-1, i, j]) else None,
            })
    return data


@app.route('/api/variables', methods=['GET'])
def list_variables():
    """Return all 3-D variables in the uploaded NetCDF file."""
    nc_path = APP_STATE.get('nc_path')
    if not nc_path:
        return jsonify({'error': 'No NetCDF file uploaded. Upload a NetCDF '
                                 'dataset first (Section 01).'}), 400
    try:
        variables = she.HeatwaveGrid.list_all_variables(nc_path)
        return jsonify({'ok': True, 'variables': variables,
                        'count': len(variables),
                        'variable_map': APP_STATE.get('variable_map') or {}})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/map_variables', methods=['POST'])
def map_variables():
    """Load the user-mapped TX / TN variables from the uploaded NetCDF.

    Body: {'tmax': <var>, 'tmin': <var>, 'primary': 'tmax'|'tmin',
           'temperature_unit': 'auto'}
    The matching grids are kept in APP_STATE['tx_grid'] / ['tn_grid'] and the
    primary one becomes APP_STATE['grid'] so every downstream computation
    (mask, detection, statistics, trends) operates on the study-area data.
    """
    req = request.json or {}
    nc_path = APP_STATE.get('nc_path')
    if not nc_path:
        return jsonify({'error': 'No NetCDF file uploaded. Upload a NetCDF '
                                 'dataset first (Section 01).'}), 400
    tmax_var = (req.get('tmax') or '').strip() or None
    tmin_var = (req.get('tmin') or '').strip() or None
    primary = req.get('primary') or 'tmax'
    temp_unit = req.get('temperature_unit', 'auto')
    if not tmax_var and not tmin_var:
        return jsonify({'error': 'Select at least one variable for TX and/or '
                                 'TN before continuing.'}), 400

    try:
        loaded = {}
        if tmax_var:
            loaded['tx_grid'] = she.HeatwaveGrid.from_netcdf_variable(
                nc_path, tmax_var, temperature_unit=temp_unit)
        if tmin_var:
            loaded['tn_grid'] = she.HeatwaveGrid.from_netcdf_variable(
                nc_path, tmin_var, temperature_unit=temp_unit)

        # Tolerate both roles pointing at the same variable
        if tmax_var and tmin_var and tmax_var == tmin_var:
            loaded['tx_grid'] = loaded['tn_grid']

        variable_map = {'tmax': tmax_var, 'tmin': tmin_var,
                        'primary': primary}
        APP_STATE['variable_map'] = variable_map
        APP_STATE['tx_grid'] = loaded.get('tx_grid')
        APP_STATE['tn_grid'] = loaded.get('tn_grid')

        # Decide the primary analysis grid
        if primary == 'tmin' and APP_STATE['tn_grid'] is not None:
            grid = APP_STATE['tn_grid']
        elif APP_STATE['tx_grid'] is not None:
            grid = APP_STATE['tx_grid']
        else:
            grid = APP_STATE['tn_grid']
        if grid is None:
            return jsonify({'error': 'No grid could be built from the mapped '
                                     'variables.'}), 400

        grid.add_noise_if_needed()
        # Re-extract the study area from the newly selected variable using the
        # uploaded shapefile, so analysis stays inside the study boundary.
        if APP_STATE.get('shp_path'):
            _apply_study_area_mask(grid)
        APP_STATE['grid'] = grid
        APP_STATE['spatial_result'] = None
        APP_STATE['trends'] = None
        APP_STATE['year_stats'] = None
        APP_STATE['year_frames'] = None

        info = _grid_json_info(grid)
        return jsonify({
            'ok': True,
            'info': info,
            'variable_map': variable_map,
            'preview': _grid_preview_payload(grid, APP_STATE.get('mask')),
            'tx_info': _grid_json_info(APP_STATE['tx_grid'])
            if APP_STATE['tx_grid'] is not None else None,
            'tn_info': _grid_json_info(APP_STATE['tn_grid'])
            if APP_STATE['tn_grid'] is not None else None,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _grid_preview_payload(grid, mask=None, points=20000):
    """Return a full spatial field + sample series for data assessment.

    ``field`` holds the complete (decimated) lat/lon grid for the most recent
    time step that is largely observed inside the study area, plus a sample
    point time series so a click or a probe can be plotted anywhere.
    """
    active = mask if mask is not None else np.ones((grid.nlat, grid.nlon), dtype=bool)

    # pick the most recent time index with observed values inside the mask
    values = grid.values
    inside = np.argwhere(active)
    t_idx = grid.ntime - 1
    if inside.size:
        sub = values[:, inside[:, 0], inside[:, 1]]
        okay = np.isfinite(sub).sum(axis=1)
        valid_t = np.where(okay > 0)[0]
        if valid_t.size:
            t_idx = int(valid_t[-1])

    field = values[t_idx].copy()
    field[~active] = np.nan

    # decimate to keep payloads manageable
    flat_valid = np.argwhere(np.isfinite(field).ravel()).size
    if flat_valid > points:
        step = int(np.ceil((grid.nlat * grid.nlon) / points))
        idx = np.s_[::step, ::step]
    else:
        idx = np.s_[:, :]
    field_lats, field_lons = grid.lats[idx[0]], grid.lons[idx[1]]
    field_z = field[idx]

    # sample point series (first observed cell near centre of the mask)
    series = []
    sample_lat = sample_lon = None
    if inside.size:
        i, j = inside[len(inside) // 2]
        sample_lat, sample_lon = float(grid.lats[i]), float(grid.lons[j])
        for time, value in zip(grid.times, values[:, i, j]):
            series.append({'date': pd.Timestamp(time).strftime('%Y-%m-%d'),
                           'value': None if np.isnan(value) else float(value)})

    return {
        'field': {
            'lats': _jsonable(field_lats.tolist()),
            'lons': _jsonable(field_lons.tolist()),
            'values': _jsonable(field_z),
            'time_idx': int(t_idx),
            'time_label': pd.Timestamp(grid.times[t_idx]).strftime('%Y-%m-%d'),
        },
        'time_series': series,
        'active_cells': int(active.sum()),
        'total_cells': int(active.size),
        'field_min': float(np.nanmin(field_z)) if np.isfinite(field_z).any() else None,
        'field_max': float(np.nanmax(field_z)) if np.isfinite(field_z).any() else None,
        'coverage': round(float(np.isfinite(field_z).sum()) / field_z.size * 100, 2)
        if field_z.size else 0.0,
        'sample_lat': sample_lat,
        'sample_lon': sample_lon,
    }


@app.route('/api/upload_shapefile', methods=['POST'])
def upload_shapefile():
    """Upload a ZIP, SHP, GeoJSON, or GJSON study-area boundary."""
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No boundary file provided.'}), 400

        filename = file.filename.lower()
        path = _save_upload(file, 'shapefile')
        gdf, source_dir = _read_boundary_file(path)
        shp_path = path if filename.endswith('.shp') else None
        if filename.endswith('.zip'):
            shp_files = _find_archive_member(source_dir, ('.shp',))
            shp_path = shp_files[0] if shp_files else None

        if gdf.crs is not None and str(gdf.crs) not in ('epsg:4326', 'EPSG:4326'):
            gdf = gdf.to_crs(epsg=4326)
        elif gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)

        bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
        names_col = None
        for c in gdf.columns:
            if str(c).lower() in ('name', 'names', 'admin_name', 'geoid', 'id'):
                names_col = c
                break

        # Build boundary polylines for plotting overlay
        from shapely.geometry import mapping
        rings = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            try:
                gi = mapping(geom)
            except Exception:
                continue
            if gi['type'] == 'Polygon':
                rings.append(list(gi['coordinates'][0]))
            elif gi['type'] == 'MultiPolygon':
                for poly in gi['coordinates']:
                    rings.append(list(poly[0]))

        boundaries = [
            {'lats': [pt[1] for pt in ring], 'lons': [pt[0] for pt in ring]}
            for ring in rings
        ]

        features = []
        for idx, row in gdf.iterrows():
            feat = {'index': int(idx)}
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.representative_point() if hasattr(geom, 'representative_point') else geom.centroid
            feat['label'] = str(row[names_col]) if names_col is not None else f'Feature {idx}'
            feat['lat'] = float(centroid.y)
            feat['lon'] = float(centroid.x)
            features.append(feat)

        APP_STATE['tmp_shapefile_dir'] = source_dir
        APP_STATE['shp_path'] = shp_path
        APP_STATE['shp_gdf_summary'] = {
            'feature_count': int(len(gdf)),
            'bounds': bounds.tolist(),
            'crs': str(gdf.crs),
            'admin_boundaries': boundaries,
            'features': features,
        }

        return jsonify(APP_STATE['shp_gdf_summary'])
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload_landmask', methods=['POST'])
def upload_landmask():
    """Upload a static land/ocean mask (GeoTIFF or NetCDF, True = land)."""
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No land/ocean mask file provided.'}), 400
        filename = file.filename.lower()
        if not filename.endswith(('.nc', '.nc4', '.netcdf', '.tif', '.tiff', '.img')):
            return jsonify({'error': 'Unsupported land mask format. Use '
                                     'GeoTIFF (.tif/.tiff) or NetCDF (.nc).'}), 400
        grid = _require_grid()
        path = _save_upload(file, 'landmask')
        mask = she.HeatwaveGrid.load_land_ocean_mask(path, grid)
        APP_STATE['land_mask_path'] = path
        APP_STATE['land_mask_static'] = np.asarray(mask, dtype=bool)
        land_cells = int(mask.sum())
        total_cells = int(mask.size)
        return jsonify({
            'ok': True,
            'filename': file.filename,
            'land_cells': land_cells,
            'total_cells': total_cells,
            'land_percent': round(land_cells / total_cells * 100, 2) if total_cells else 0.0,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """Mask the grid by shapefile / land mask and optionally fill gaps / smooth."""
    req = request.json or {}
    grid = _require_grid()
    t0 = time.time()

    report = {'steps': [], 'masked_cells': 0, 'total_cells': grid.nlat * grid.nlon}
    work = grid.copy()
    mask = None

    use_shapefile = bool(req.get('mask_with_shapefile', False))
    if use_shapefile:
        shp = APP_STATE.get('shp_path')
        if not shp:
            return jsonify({'error': 'No shapefile loaded. Upload a zipped '
                                     'shapefile in Section 01 first.'}), 400
        mask = work.mask_with_shapefile(shp)
        work.apply_mask(mask)
        APP_STATE['mask'] = mask
        report['steps'].append('Masked grid with shapefile boundary.')
    else:
        # fallback: simple bounding box
        bounds = req.get('bounds')
        if bounds and len(bounds) == 4:
            mask = work.build_mask_from_latlon(bounds)
            work.apply_mask(mask)
            APP_STATE['mask'] = mask
            report['steps'].append('Masked grid with manually supplied bounds.')

    # Land / ocean exclusion: keep cells with valid data (plus any uploaded
    # static land mask), so sea cells are excluded from all computations.
    exclude_sea = bool(req.get('exclude_sea', False))
    if exclude_sea:
        try:
            mcov = float(req.get('min_time_coverage', 0.05))
        except (TypeError, ValueError):
            mcov = 0.05
        land = work.build_land_mask(min_time_coverage=mcov)
        static = APP_STATE.get('land_mask_static')
        if static is not None and static.shape == land.shape:
            land = land & static
            report['steps'].append(
                'Combined automatic land mask with uploaded static land/'
                'ocean mask (True = land).')
        work.apply_mask(land)
        if mask is None:
            mask = land
            APP_STATE['mask'] = mask
        else:
            combined = mask & land
            work.apply_mask(combined)
            mask = combined
            APP_STATE['mask'] = combined
        report['steps'].append(
            f'Excluded ocean/sea cells (auto land mask, cells kept when >= '
            f'{mcov:.2f} of time steps are valid).')
        report['excluded_sea'] = True

    if mask is None:
        mask = np.ones((work.nlat, work.nlon), dtype=bool)
        APP_STATE['mask'] = mask
        report['steps'].append('No mask applied (using full domain).')

    report['masked_cells'] = int(mask.sum())
    report['masked_percent'] = round(float(mask.sum() / mask.size * 100), 2)

    fill_gaps = bool(req.get('fill_gaps', False))
    if fill_gaps:
        n_before = int(np.isnan(work.values).sum())
        work.values = _fill_grid_gaps(work.values)
        n_after = int(np.isnan(work.values).sum())
        report['steps'].append(
            f'Gap filling applied: {n_before - n_after} NaN(s) interpolated '
            f'({n_after} NaN(s) remain).')

    smooth = bool(req.get('smooth', False))
    if smooth:
        window = int(req.get('smooth_window', 7))
        work.values = _time_smooth_grid(work.values, window)
        report['steps'].append(f'Temporal smoothing applied (window={window} days).')

    zscore = bool(req.get('standardize', False))
    if zscore:
        mean = np.nanmean(work.values)
        std = np.nanstd(work.values)
        work.values = (work.values - mean) / (std if std > 0 else 1)
        report['steps'].append('Data standardised to z-scores.')

    APP_STATE['grid'] = work
    APP_STATE['spatial_result'] = None
    report['ok'] = True
    report['grid_info'] = _grid_json_info(work)
    report['preview'] = _grid_preview_payload(work, APP_STATE.get('mask'))
    report['elapsed_ms'] = int((time.time() - t0) * 1000)
    return jsonify(report)


@app.route('/api/preview/field', methods=['POST'])
def preview_field():
    """Return the observed spatial field for assessment (time step or mean)."""
    grid = APP_STATE.get('grid')
    if grid is None:
        return jsonify({'ok': False, 'error': 'No data loaded.'}), 400
    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}
    mode = payload.get('mode', 'last')
    try:
        t_idx = int(payload.get('time_idx', -1))
    except (TypeError, ValueError):
        t_idx = -1

    mask = APP_STATE.get('mask')
    active = mask if mask is not None else np.ones((grid.nlat, grid.nlon), dtype=bool)

    values = grid.values
    if mode == 'mean':
        field = np.nanmean(values, axis=0) if grid.ntime else np.full((grid.nlat, grid.nlon), np.nan)
        label = 'Time mean of loaded period'
    elif mode == 'index' and 0 <= t_idx < grid.ntime:
        field = values[t_idx].copy()
        label = pd.Timestamp(grid.times[t_idx]).strftime('%Y-%m-%d')
    else:
        inside = np.argwhere(active)
        t_idx = grid.ntime - 1
        if inside.size:
            sub = values[:, inside[:, 0], inside[:, 1]]
            okay = np.isfinite(sub).sum(axis=1)
            valid_t = np.where(okay > 0)[0]
            if valid_t.size:
                t_idx = int(valid_t[-1])
        field = values[t_idx].copy()
        label = pd.Timestamp(grid.times[t_idx]).strftime('%Y-%m-%d')

    field = np.where(active, field, np.nan).astype(float)
    try:
        max_pts = int(payload.get('max_points', 20000))
    except (TypeError, ValueError):
        max_pts = 20000
    flat_valid = np.isfinite(field).sum()
    if flat_valid > max_pts:
        step = int(np.ceil((grid.nlat * grid.nlon) / max_pts))
        idx = np.s_[::step, ::step]
    else:
        idx = np.s_[:, :]

    vals = _jsonable(field[idx])
    fin = np.isfinite(field[idx])
    return jsonify({
        'ok': True,
        'field': {
            'lats': _jsonable(grid.lats[idx[0]].tolist()),
            'lons': _jsonable(grid.lons[idx[1]].tolist()),
            'values': vals,
            'time_idx': int(t_idx),
            'time_label': label,
        },
        'active_cells': int(active.sum()),
        'total_cells': int(active.size),
        'field_min': float(np.min(field[idx][fin])) if fin.any() else None,
        'field_max': float(np.max(field[idx][fin])) if fin.any() else None,
        'coverage': round(float(fin.sum()) / fin.size * 100, 2) if fin.size else 0.0,
    })


def _fill_grid_gaps(values):
    """Linearly interpolate missing values along the time axis (vectorised)."""
    nt, nlat, nlon = values.shape
    out = values.copy()
    flat = out.reshape(nt, -1)
    all_nan = ~np.isfinite(flat).any(axis=0)
    if all_nan.all():
        return out
    df = pd.DataFrame(flat.astype(np.float32))
    df = df.interpolate(method='linear', axis=0, limit_area='inside')
    df = df.ffill().bfill()
    filled = df.to_numpy(dtype=np.float32)
    filled[:, all_nan] = np.nan
    out[:] = filled.reshape(nt, nlat, nlon)
    return out


def _time_smooth_grid(values, window=7):
    """Simple unweighted rolling average along the time axis (vectorised)."""
    nt, nlat, nlon = values.shape
    kernel_size = max(1, int(window))
    flat = values.reshape(nt, -1).astype(np.float32)
    all_nan = ~np.isfinite(flat).any(axis=0)
    work = flat.copy()
    if not all_nan.all():
        df = pd.DataFrame(work)
        df = df.interpolate(method='linear', axis=0, limit_area='inside')
        df = df.ffill().bfill()
        work = df.to_numpy(dtype=np.float32)
    work[:, all_nan] = np.nan
    valid_cols = ~all_nan
    if valid_cols.any() and kernel_size > 1:
        try:
            from scipy.ndimage import uniform_filter1d
            smoothed = uniform_filter1d(work, size=kernel_size, axis=0,
                                        mode='constant', cval=0.0)
            work[:, valid_cols] = smoothed[:, valid_cols]
        except Exception:
            # fallback: numpy convolution column-by-column (slow but safe)
            kernel = np.ones(kernel_size) / kernel_size
            cols = np.where(valid_cols)[0]
            for c in cols:
                series = work[:, c]
                work[:, c] = np.convolve(series, kernel, mode='same')
    out = values.copy()
    out[:] = work.reshape(nt, nlat, nlon)
    return out


@app.route('/api/data_info', methods=['GET'])
def data_info():
    grid = APP_STATE.get('grid')
    if grid is None:
        return jsonify({'loaded': False, 'error': 'No data loaded.'})
    return jsonify({'loaded': True, 'info': _grid_json_info(grid),
                    'mask_applied': bool(APP_STATE.get('mask') is not None),
                    'masked_cells': int(np.count_nonzero(APP_STATE['mask']))
                        if APP_STATE.get('mask') is not None else None})


# ----------------------------------------------------------------------------
#  02. HEATWAVE COMPUTATIONS
# ----------------------------------------------------------------------------

@app.route('/api/detect', methods=['POST'])
def compute_detect():
    """Launch spatial heatwave detection as a background job.

    Returns immediately with job metadata; progress/result are read by
    polling ``GET /api/status``.
    """
    req = request.json or {}
    grid = _require_grid()

    with _COMPUTE_LOCK:
        if COMPUTE_STATE.get('running'):
            return jsonify({'error': 'A heatwave computation is already '
                                     'running. Stop it or wait for it to '
                                     'finish.'}), 409

    try:
        clim_start = int(req.get('clim_start', 1981))
        clim_end = int(req.get('clim_end', 2010))
    except (TypeError, ValueError):
        return jsonify({'error': 'Climatology years must be integers.'}), 400
    pctile = float(req.get('pctile', 90))
    min_duration = int(req.get('min_dur', 3))
    join_across_gaps = bool(req.get('join_gaps', False))
    max_gap = int(req.get('max_gap', 2))
    smooth_percentile = bool(req.get('smooth_pctile', True))
    smooth_pctile_width = int(req.get('smooth_pctile_width', 31))
    window_half_width = int(req.get('window_half_width', 5))
    n_jobs = _requested_cores(req.get('n_jobs', 0))

    mask = None
    if APP_STATE.get('mask') is not None:
        mask = np.asarray(APP_STATE['mask'], dtype=bool)
    total_cells = int(mask.sum()) if mask is not None else grid.nlat * grid.nlon
    if total_cells == 0:
        return jsonify({'error': 'The study-area mask contains no active '
                                 'cells. Check the shapefile / land mask.'}), 400

    params = {
        'grid': grid,
        'mask': mask,
        'n_jobs': n_jobs,
        'clim_start': clim_start,
        'clim_end': clim_end,
        'pctile': pctile,
        'min_duration': min_duration,
        'join_across_gaps': join_across_gaps,
        'max_gap': max_gap,
        'smooth_percentile': smooth_percentile,
        'smooth_pctile_width': smooth_pctile_width,
        'window_half_width': window_half_width,
    }

    with _COMPUTE_LOCK:
        COMPUTE_STATE.update(
            running=True, phase='starting', done=0, total=total_cells,
            percent=0.0, label='Starting detection ...', error=None,
            cancelled=False, cancel_requested=False,
            started_at=time.time(), finished_at=None, elapsed_ms=0,
            n_jobs=n_jobs, masked_cells=total_cells, last_summary=None)

    threading.Thread(target=_detect_worker, args=(params,), daemon=True).start()

    return jsonify({
        'started': True,
        'job': 'detect',
        'n_jobs': n_jobs,
        'masked_cells': total_cells,
        'mask': bool(mask is not None),
    })


@app.route('/api/status', methods=['GET'])
def compute_status():
    """Poll detection progress (percent / elapsed / phase / result)."""
    with _COMPUTE_LOCK:
        snapshot = dict(COMPUTE_STATE)
    return jsonify(snapshot)


@app.route('/api/cores', methods=['GET'])
def cores_info():
    return jsonify({'total': multiprocessing.cpu_count(),
                    'max': multiprocessing.cpu_count()})


@app.route('/api/stop', methods=['POST'])
def stop_compute():
    """Request cancellation of the running detection job."""
    with _COMPUTE_LOCK:
        running = bool(COMPUTE_STATE.get('running'))
        if running:
            COMPUTE_STATE['cancel_requested'] = True
            COMPUTE_STATE['label'] = 'Cancellation requested ...'
    return jsonify({'ok': True, 'cancelling': running})


@app.route('/api/clear', methods=['POST'])
def clear_all():
    """Stop any job and wipe uploaded data, masks, results and caches."""
    with _COMPUTE_LOCK:
        if COMPUTE_STATE.get('running'):
            COMPUTE_STATE['cancel_requested'] = True
    for key in list(APP_STATE.keys()):
        APP_STATE[key] = ({} if key == 'tmp_geo' else None)
    # Wipe persisted uploads / caches on disk
    for name in os.listdir(UPLOAD_DIR):
        target = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isdir(target):
                import shutil
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(target)
        except Exception:
            pass
    with _COMPUTE_LOCK:
        COMPUTE_STATE.update(
            running=False, phase='idle', done=0, total=0, percent=0.0,
            label='State cleared.', elapsed_ms=0, started_at=None,
            finished_at=None, cancel_requested=False, cancelled=False,
            error=None, last_summary=None)
    return jsonify({'ok': True})


@app.route('/api/statistics', methods=['POST'])
def compute_statistics():
    """Regional statistics plus annual block statistics for the region."""
    req = request.json or {}
    result = _require_result()
    grid = _require_grid()

    year_start = req.get('year_start')
    year_end = req.get('year_end')
    block_length = int(req.get('block_length', 1))

    if year_start is None:
        year_start = int(grid.times.year.min())
    if year_end is None:
        year_end = int(grid.times.year.max())

    year_stats = she.build_annual_field_stats(
        grid, result, year_start=year_start, year_end=year_end,
        block_length=block_length)
    APP_STATE['year_stats'] = year_stats

    # --- annual region totals (sum of all-cell values) ---
    region_annual = []
    mask = APP_STATE.get('mask')
    for (b0, b1) in year_stats['blocks']:
        cells = [(i, j, b0) for (i, j, b) in year_stats['count'].keys() if b == b0]
        total_events = sum(year_stats['count'].get(c, 0) for c in cells)
        total_days = sum(year_stats['days'].get(c, 0) for c in cells)
        total_cumint = sum(year_stats['cumint'].get(c, 0) for c in cells)
        active_cells = len(cells)
        region_annual.append({
            'block_start': b0,
            'block_end': b1,
            'active_cells': active_cells,
            'total_events': float(total_events),
            'total_hw_days': float(total_days),
            'total_cumulative_intensity': float(total_cumint),
            'mean_events_per_cell': float(total_events / active_cells) if active_cells else 0,
            'mean_hw_days_per_cell': float(total_days / active_cells) if active_cells else 0,
        })

    return jsonify({
        'ok': True,
        'summary': result.summarize(),
        'regional_annual': region_annual,
        'blocks': year_stats['blocks'],
        'grid_info': _grid_json_info(grid),
    })


@app.route('/api/trends', methods=['POST'])
def compute_trends():
    """Compute spatial trend fields (linear regression or Mann-Kendall)."""
    req = request.json or {}
    result = _require_result()
    grid = _require_grid()

    metric = req.get('metric', 'n_events')
    method = req.get('method', 'linear')
    clim_start = req.get('clim_start') or int(grid.times.year.min())
    clim_end = req.get('clim_end') or int(grid.times.year.max())

    if method == 'mannkendall':
        trends = she.mann_kendall_field(grid, result, metric,
                                        clim_start=clim_start,
                                        clim_end=clim_end)
        APP_STATE['trends'] = trends
        # Build maps for plotting
        tau_arr = np.full((grid.nlat, grid.nlon), np.nan)
        p_arr = np.full((grid.nlat, grid.nlon), np.nan)
        for (i, j), v in trends['tau'].items():
            tau_arr[i, j] = v
        for (i, j), v in trends['pvalue'].items():
            p_arr[i, j] = v
        return jsonify({
            'ok': True, 'method': 'mann-kendall', 'metric': metric,
            'years': [int(y) for y in trends['years']],
            'tau': _matrix_payload(tau_arr),
            'pvalue': _matrix_payload(p_arr),
            'summary': _trend_summary(tau_arr, p_arr),
        })
    else:
        trends = she.compute_spatial_trends(
            grid, result, metric=metric, clim_start=clim_start,
            clim_end=clim_end)
        APP_STATE['trends'] = trends
        slope_arr = np.full((grid.nlat, grid.nlon), np.nan)
        p_arr = np.full((grid.nlat, grid.nlon), np.nan)
        r2_arr = np.full((grid.nlat, grid.nlon), np.nan)
        for (i, j), v in trends['slope'].items():
            slope_arr[i, j] = v
        for (i, j), v in trends['pvalue'].items():
            p_arr[i, j] = v
        for (i, j), v in trends['r2'].items():
            r2_arr[i, j] = v
        return jsonify({
            'ok': True, 'method': 'linear-regression', 'metric': metric,
            'years': [int(y) for y in trends['years']],
            'slope': _matrix_payload(slope_arr),
            'pvalue': _matrix_payload(p_arr),
            'r2': _matrix_payload(r2_arr),
            'summary': _trend_summary(slope_arr, p_arr),
        })


def _jsonable(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (bool, str, int, float)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if np.ma.isMaskedArray(obj):
        return _jsonable(np.ma.filled(obj, fill_value=np.nan).tolist())
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _fig_to_json(fig):
    """Convert a plotly Figure to JSON with plain arrays (no bdata compression).

    plotly 6.x encodes numeric arrays as {dtype:'f8', bdata:'<base64>'} and
    leaves numpy / pandas datetime arrays as raw numpy objects. Browser plotly
    may not decode these, so we convert everything to standard JSON here.
    """
    import base64, struct as _struct, math as _math

    def _fmt_time(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat(sep=' ')
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, np.datetime64):
            return str(np.datetime_as_string(obj, unit='D'))
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    def _decode(obj):
        if isinstance(obj, dict) and 'bdata' in obj and 'dtype' in obj:
            raw = base64.b64decode(obj['bdata'])
            dt = obj['dtype']
            shape_str = obj.get('shape')
            n = len(raw)
            if dt == 'f8':
                vals = list(_struct.unpack(f'<{n // 8}d', raw))
                vals = [None if _math.isnan(v) else v for v in vals]
            elif dt == 'f4':
                vals = list(_struct.unpack(f'<{n // 4}f', raw))
                vals = [None if _math.isnan(v) else v for v in vals]
            elif dt == 'i4':
                vals = list(_struct.unpack(f'<{n // 4}i', raw))
            elif dt == 'i2':
                vals = list(_struct.unpack(f'<{n // 2}h', raw))
            elif dt == 'b1':
                vals = list(_struct.unpack(f'<{n}B', raw))
            else:
                vals = list(raw)
            # restore shape for 2-D heatmap / contour arrays
            if shape_str and ',' in shape_str:
                dims = [int(d) for d in shape_str.split(',')]
                if len(dims) == 2:
                    vals = [vals[r * dims[1]:(r + 1) * dims[1]] for r in range(dims[0])]
            return vals
        if isinstance(obj, dict):
            return {k: _decode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_decode(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return _decode(obj.tolist())
        if isinstance(obj, np.datetime64):
            return _fmt_time(obj)
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return _fmt_time(obj)
        if isinstance(obj, float) and _math.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if not _math.isfinite(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    return json.dumps(_decode(fig.to_dict()), default=str)


def _matrix_payload(arr):
    if APP_STATE.get('grid') is None:
        return {'lats': [], 'lons': [], 'values': []}
    g = APP_STATE['grid']
    return {'lats': g.lats.tolist(), 'lons': g.lons.tolist(),
            'values': _jsonable(arr)}


def _trend_summary(values, pvalues):
    valid_p = pvalues[~np.isnan(pvalues)]
    sig = valid_p[valid_p < 0.05]
    vals = values[~np.isnan(values)]
    return {
        'cells': int(len(vals)),
        'positive': int(np.sum(vals > 0)),
        'negative': int(np.sum(vals < 0)),
        'no_trend': int(np.sum(vals == 0)),
        'significant_cells_5pct': int(len(sig)),
        'significant_frac': float(len(sig) / len(valid_p)) if len(valid_p) else 0,
        'mean_value': float(np.nanmean(vals)) if len(vals) else None,
    }


@app.route('/api/geojson_boundaries', methods=['GET'])
def geojson_boundaries():
    """Return the uploaded boundary rings for map overlays."""
    summary = APP_STATE.get('shp_gdf_summary')
    if not summary:
        return jsonify({'boundaries': []})
    return jsonify({'boundaries': summary['admin_boundaries'],
                    'features': summary['features']})


# ----------------------------------------------------------------------------
#  03. VISUALIZATION
# ----------------------------------------------------------------------------

def _iso_dates(times):
    """Convert a pandas DatetimeIndex or numpy array to plain ISO date strings.

    Using ISO strings (rather than letting plotly encode datetime64 to epoch ns
    or ms) avoids browser/plotly.js ms-vs-ns ambiguity and guarantees the
    frontend interprets the axis as dates.
    """
    if hasattr(times, 'tolist'):
        out = []
        for t in times.tolist():
            if hasattr(t, 'strftime'):
                out.append(t.strftime('%Y-%m-%d'))
            else:
                out.append(str(t)[:10])
        return out
    return [str(t)[:10] for t in times]

@app.route('/api/viz/metric_map', methods=['POST'])
def viz_metric_map():
    """Heatmap of a single metric over the spatial domain."""
    req = request.json or {}
    result = _require_result()
    metric = req.get('metric', 'n_events')

    if metric not in result.metric_maps:
        return jsonify({'error': f'Metric "{metric}" not available.'}), 400

    arr = np.full((result.grid.nlat, result.grid.nlon), np.nan)
    for (i, j), v in result.metric_maps[metric].items():
        arr[i, j] = v

    titles = {
        'n_events': 'Total Heatwave Events',
        'mean_duration': 'Mean Heatwave Duration',
        'max_duration': 'Maximum Heatwave Duration',
        'mean_intensity_max': 'Mean Peak Intensity',
        'max_intensity_max': 'Maximum Peak Intensity',
        'mean_intensity_mean': 'Mean Intensity',
        'cumulative_intensity': 'Cumulative Intensity',
        'total_hw_days': 'Total Heatwave Days',
        'mean_year_freq': 'Mean Events Per Year',
        'max_intensity_abs': 'Maximum Absolute Intensity',
        'categories_count': 'Max Event Category',
    }
    colorscales = {
        'n_events': 'YlOrRd',
        'mean_duration': 'YlOrRd',
        'max_duration': 'YlOrRd',
        'mean_intensity_max': 'RdYlBu_r',
        'max_intensity_max': 'RdYlBu_r',
        'mean_intensity_mean': 'RdYlBu_r',
        'cumulative_intensity': 'Hot',
        'total_hw_days': 'YlOrRd',
        'mean_year_freq': 'YlOrRd',
        'max_intensity_abs': 'RdYlBu_r',
        'categories_count': 'Viridis',
    }
    fig = _field_figure(
        arr, result.grid.lats, result.grid.lons,
        titles.get(metric, metric), colorscale=colorscales.get(metric, 'RdYlBu_r'),
        label=metric, unit=('events' if metric == 'n_events'
                            else 'days' if metric in ('mean_duration',
                                                       'max_duration',
                                                       'total_hw_days')
                            else '\u00b0C' if 'intensity' in metric
                            else ''))

    # Significant cells marker layer if trends available
    if req.get('show_significant') and APP_STATE.get('trends'):
        trends = APP_STATE['trends']
        lats, lons, pvals = [], [], []
        for (i, j), p in trends.get('pvalue', {}).items():
            if p < 0.05:
                lats.append(float(result.grid.lats[i]))
                lons.append(float(result.grid.lons[j]))
                pvals.append(p)
        if lats:
            fig.add_trace(go.Scatter(
                x=lons, y=lats, mode='markers', name='significant (p<0.05)',
                marker=dict(size=5, symbol='circle', color='#00e5cc',
                            line=dict(color='white', width=0.7)),
                hovertemplate='Lon: %{x:.3f}<br>Lat: %{y:.3f}<extra></extra>',
                showlegend=True))

    return jsonify({'ok': True, 'plot': _fig_to_json(fig)})


@app.route('/api/viz/timeseries', methods=['POST'])
def viz_timeseries():
    """Full detection time series for a chosen grid point."""
    req = request.json or {}
    result = _require_result()
    grid = _require_grid()

    lat = float(req.get('lat'))
    lon = float(req.get('lon'))

    i = int(np.argmin(np.abs(grid.lats - lat)))
    j = int(np.argmin(np.abs(grid.lons - lon)))

    if (i, j) not in result.mhws_map:
        series = grid.values[:, i, j]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=_iso_dates(grid.times), y=series, mode='lines', name='Temperature',
            line=dict(color='#4fc3f7', width=1.3)))
        fig.update_layout(template='plotly_dark',
                          paper_bgcolor='#0c1714', plot_bgcolor='#070e0c',
                          title=dict(text=f'<b>Temperature time series '
                                          f'({lat:.3f}\u00b0N, {lon:.3f}\u00b0E)</b>'),
                          height=430)
        return jsonify({'ok': True, 'plot': _fig_to_json(fig),
                        'no_events': True, 'closest_lat': float(grid.lats[i]),
                        'closest_lon': float(grid.lons[j])})

    mh_res, clim_res = result.mhws_map[(i, j)]
    t_ord = grid.t_ordinal()
    series = grid.values[:, i, j]
    iso = _iso_dates(grid.times)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iso, y=clim_res['seas'], mode='lines',
        name='Climatology', line=dict(color='#66bb6a', dash='dot', width=1.0)))
    fig.add_trace(go.Scatter(
        x=iso, y=clim_res['thresh'], mode='lines',
        name='Threshold', line=dict(color='#ffa726', dash='dash', width=1.2)))
    fig.add_trace(go.Scatter(
        x=iso, y=series, mode='lines', name='Temperature',
        line=dict(color='#4fc3f7', width=1.3)))

    if mh_res.get('n_events', 0) > 0:
        for k in range(mh_res['n_events']):
            s = np.where(t_ord == mh_res['time_start'][k])[0][0]
            e = np.where(t_ord == mh_res['time_end'][k])[0][0]
            fig.add_trace(go.Scatter(
                x=iso[s:e + 1], y=clim_res['thresh'][s:e + 1],
                mode='lines', line=dict(width=0), hoverinfo='skip',
                showlegend=False))
            fig.add_trace(go.Scatter(
                x=iso[s:e + 1], y=series[s:e + 1],
                mode='lines', line=dict(color='#ef5350', width=1.8),
                fill='tonexty', fillcolor='rgba(239,83,80,0.45)',
                name='Heatwave',
                showlegend=(k == 0)))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0c1714', plot_bgcolor='#070e0c',
        title=dict(text=f'<b>Point time series</b> \u2014 '
                        f'({float(grid.lats[i]):.3f}\u00b0N, '
                        f'{float(grid.lons[j]):.3f}\u00b0E)<br>'
                        f'<span style="font-size:11px;color:#aaa;">'
                        f'{mh_res.get("n_events", 0)} heatwave event(s) detected'
                        f'</span>', font=dict(size=14)),
        xaxis_title='Date', yaxis_title='Temperature (\u00b0C)',
        hovermode='x unified', legend=dict(orientation='h', y=-0.25),
        height=460, margin=dict(l=60, r=30, t=60, b=60))

    events = []
    for k in range(mh_res.get('n_events', 0)):
        events.append({
            'start': str(mh_res['date_start'][k]),
            'peak': str(mh_res['date_peak'][k]),
            'end': str(mh_res['date_end'][k]),
            'duration': int(mh_res['duration'][k]),
            'intensity_max': round(float(mh_res['intensity_max'][k]), 3),
            'intensity_mean': round(float(mh_res['intensity_mean'][k]), 3),
            'intensity_cumulative': round(float(mh_res['intensity_cumulative'][k]), 3),
            'category': str(mh_res['category'][k]),
        })

    return jsonify({
        'ok': True, 'plot': _fig_to_json(fig),
        'closest_lat': float(grid.lats[i]), 'closest_lon': float(grid.lons[j]),
        'events': events, 'no_events': mh_res.get('n_events', 0) == 0,
    })


@app.route('/api/viz/animation', methods=['POST'])
def viz_animation():
    """Animated spatial view of a metric across years/blocks."""
    req = request.json or {}
    result = _require_result()
    grid = _require_grid()
    year_stats = APP_STATE.get('year_stats')

    if year_stats is None:
        year_start = req.get('year_start') or int(grid.times.year.min())
        year_end = req.get('year_end') or int(grid.times.year.max())
        year_stats = she.build_annual_field_stats(
            grid, result, year_start=year_start, year_end=year_end)
        APP_STATE['year_stats'] = year_stats

    metric = req.get('metric', 'count')
    blocks = year_stats['blocks']
    key = metric if metric in ('count', 'days', 'cumint', 'imax', 'freq') else 'count'

    lats = grid.lats.tolist()
    lons = grid.lons.tolist()

    frames = []
    steps = []
    for idx, (b0, b1) in enumerate(blocks):
        arr = np.full((grid.nlat, grid.nlon), np.nan)
        cell_map = year_stats[key]
        for (i, j, b), v in cell_map.items():
            if b == b0:
                arr[i, j] = v
        frame = go.Frame(
            data=[go.Heatmap(x=lons, y=lats, z=arr.tolist(),
                             zmin=0, colorscale='YlOrRd')],
            name=str(b0),
            layout=dict(title='<b>' + key + ' per cell \u2014 '
                              + str(b0) + '\u2013' + str(b1) + '</b>'))
        frames.append(frame)
        step = dict(method='animate', args=[
            [str(b0)], {'frame': {'duration': 600, 'redraw': True},
                        'mode': 'immediate'}],
            label=str(b0))
        steps.append(step)

    play_btn = dict(label='Play', method='animate', args=[
        [None], {'frame': {'duration': 600, 'redraw': True},
                 'fromcurrent': True}])
    pause_btn = dict(label='Pause', method='animate', args=[
        [None], {'frame': {'duration': 0, 'redraw': False},
                 'mode': 'immediate'}])

    layout = go.Layout(
        template='plotly_dark',
        paper_bgcolor='#0c1714',
        plot_bgcolor='#070e0c',
        height=520,
        title=dict(text='<b>Spatial dynamics of heatwave metrics over time</b>'),
        xaxis_title='Longitude (\u00b0E)',
        yaxis_title='Latitude (\u00b0N)',
        margin=dict(l=60, r=30, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor='x'),
        sliders=[dict(active=0, steps=steps, x=0.1, len=0.8,
                      pad=dict(t=40),
                      currentvalue=dict(prefix='Year: ', visible=True,
                                        xanchor='center'))],
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[play_btn, pause_btn],
                          x=0.05, y=0.0)],
    )

    fig = go.Figure(
        data=[go.Heatmap(x=lons, y=lats,
                         z=np.full((grid.nlat, grid.nlon), np.nan),
                         zmin=0, colorscale='YlOrRd', name=key)],
        layout=layout,
    )
    fig.frames = frames
    return jsonify({'ok': True, 'plot': _fig_to_json(fig)})


@app.route('/api/viz/annual_bars', methods=['POST'])
def viz_annual_bars():
    """Regional annual totals as a bar chart (all active cells summed)."""
    req = request.json or {}
    result = _require_result()
    year_stats = APP_STATE.get('year_stats')

    if year_stats is None:
        grid = _require_grid()
        year_stats = she.build_annual_field_stats(grid, result)
        APP_STATE['year_stats'] = year_stats

    labels = [f'{b0}\u2013{b1}' if b0 != b1 else str(b0)
              for (b0, b1) in year_stats['blocks']]

    counts = []
    days = []
    cumints = []
    for (b0, b1) in year_stats['blocks']:
        c00 = [(i, j, b0) for (i, j, b) in year_stats['count'].keys() if b == b0]
        counts.append(sum(year_stats['count'].get(c, 0) for c in c00))
        days.append(sum(year_stats['days'].get(c, 0) for c in c00))
        cumints.append(sum(year_stats['cumint'].get(c, 0) for c in c00))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, name='Events',
                         marker_color='#42a5f5'))
    fig.add_trace(go.Bar(x=labels, y=days, name='HW Days',
                         marker_color='#ffa726'))
    fig.add_trace(go.Bar(x=labels, y=cumints, name='Cumulative Intensity',
                         marker_color='#ef5350', visible='legendonly'))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0c1714', plot_bgcolor='#070e0c',
        title=dict(text='<b>Regional annual heatwave totals</b> '
                        '(all cells summed)'),
        barmode='group', xaxis_title='Year', yaxis_title='Value',
        hovermode='x unified', legend=dict(orientation='h', y=-0.3),
        height=400, margin=dict(l=60, r=30, t=55, b=60))

    return jsonify({'ok': True, 'plot': _fig_to_json(fig)})


# ----------------------------------------------------------------------------
#  04. COMPARATIVE ANALYSIS
# ----------------------------------------------------------------------------

@app.route('/api/comparison/upload_secondary', methods=['POST'])
def upload_secondary():
    """Load a second dataset (for dataset-vs-dataset comparison)."""
    try:
        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No secondary file provided.'}), 400
        filename = file.filename.lower()
        if filename.endswith('.nc'):
            path = _save_upload(file, 'data2')
            grid = she.HeatwaveGrid.from_netcdf(path)
            source = 'netcdf'
        elif filename.endswith(('.tif', '.tiff')):
            path = _save_upload(file, 'data2')
            grid = she.HeatwaveGrid.from_geotiff_stack([path])
            source = 'geotiff'
        elif filename.endswith('.zip'):
            path = _save_upload(file, 'data2')
            with zipfile.ZipFile(path, 'r') as zf:
                tiffs = [n for n in zf.namelist()
                         if n.lower().endswith(('.tif', '.tiff'))]
                if not tiffs:
                    raise ValueError('No GeoTIFFs in archive.')
                ex = os.path.join(os.path.dirname(path), 'tiffs2')
                os.makedirs(ex, exist_ok=True)
                zf.extractall(ex)
                tiff_paths = [os.path.join(ex, n) for n in tiffs]
            grid = she.HeatwaveGrid.from_geotiff_stack(tiff_paths)
            source = 'geotiff_zip'
        else:
            return jsonify({'error': 'Unsupported secondary format.'}), 400
        grid.add_noise_if_needed()
        APP_STATE['secondary_grid'] = grid
        APP_STATE['secondary_result'] = None
        return jsonify({'ok': True, 'source': source,
                        'info': _grid_json_info(grid)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/comparison/compare', methods=['POST'])
def comparison_compare():
    """Compare two fields (mode='period' or mode='dataset')."""
    req = request.json or {}
    result_a = _require_result()
    grid_a = _require_grid()

    mode = req.get('mode', 'period')

    clim_start_a = int(req.get('clim_start_a', result_a.params.get('clim_start', 1981)))
    clim_end_a = int(req.get('clim_end_a', result_a.params.get('clim_end', 2010)))
    pctile = float(req.get('pctile', 90))
    min_duration = int(req.get('min_dur', 3))
    join_gaps = bool(req.get('join_gaps', False))
    max_gap = int(req.get('max_gap', 2))

    grid_b = None
    result_b = None
    label_a = 'Field A (current)'
    label_b = 'Field B'

    if mode == 'dataset':
        grid_b = APP_STATE.get('secondary_grid')
        if grid_b is None:
            return jsonify({'error': 'No secondary dataset loaded. Upload one '
                                     'in the Comparative Analysis section.'}), 400
        if grid_b.shape[1:] != grid_a.shape[1:]:
            return jsonify({'error': 'Secondary dataset grid shape must match '
                                     'the primary grid (spatial dims).'}), 400
        res_b = she.run_spatial_detection(
            grid_b, clim_start=clim_start_a, clim_end=clim_end_a,
            pctile=pctile, min_duration=min_duration,
            join_across_gaps=join_gaps, max_gap=max_gap,
            mask=APP_STATE.get('mask'))
        APP_STATE['secondary_result'] = res_b
        result_b = res_b
        label_b = f'Field B (secondary, {grid_b.variable})'
    else:
        # period mode: re-run detection A for period A, then for period B
        clim_start_b = int(req.get('clim_start_b', 2001))
        clim_end_b = int(req.get('clim_end_b', 2020))
        res_a = she.run_spatial_detection(
            grid_a, clim_start=clim_start_a, clim_end=clim_end_a,
            pctile=pctile, min_duration=min_duration,
            join_across_gaps=join_gaps, max_gap=max_gap,
            mask=APP_STATE.get('mask'))
        APP_STATE['spatial_result'] = res_a
        res_b = she.run_spatial_detection(
            grid_a, clim_start=clim_start_b, clim_end=clim_end_b,
            pctile=pctile, min_duration=min_duration,
            join_across_gaps=join_gaps, max_gap=max_gap,
            mask=APP_STATE.get('mask'))
        result_a, result_b = res_a, res_b
        label_a = f'Period A ({clim_start_a}\u2013{clim_end_a})'
        label_b = f'Period B ({clim_start_b}\u2013{clim_end_b})'

    stats_table = she.grid_stats_table(result_a, result_b)
    diff_maps = {}
    for metric in ('n_events', 'total_hw_days', 'mean_intensity_max',
                   'mean_duration', 'cumulative_intensity', 'mean_year_freq'):
        arr_a = np.full((grid_a.nlat, grid_a.nlon), np.nan)
        arr_b = np.full((grid_a.nlat, grid_a.nlon), np.nan)
        for (i, j), v in result_a.metric_maps.get(metric, {}).items():
            arr_a[i, j] = v
        for (i, j), v in result_b.metric_maps.get(metric, {}).items():
            arr_b[i, j] = v
        diff_maps[metric] = {
            'lats': grid_a.lats.tolist(),
            'lons': grid_a.lons.tolist(),
            'a': _jsonable(arr_a), 'b': _jsonable(arr_b),
            'diff': _jsonable(arr_a - arr_b),
        }

    return jsonify({
        'ok': True,
        'mode': mode,
        'label_a': label_a,
        'label_b': label_b,
        'stats_table': stats_table,
        'diff_maps': diff_maps,
        'a_summary': result_a.summarize(),
        'b_summary': result_b.summarize(),
    })


# ----------------------------------------------------------------------------
#  05. EXPORT
# ----------------------------------------------------------------------------

@app.route('/api/export/fields', methods=['POST'])
def export_fields():
    """Export metric arrays (and optionally trends) as CSV/JSON/NetCDF."""
    req = request.json or {}
    result = _require_result()
    grid = _require_grid()
    fmt = req.get('format', 'csv')

    metrics = req.get('metrics') or ['n_events', 'mean_duration',
                                     'mean_intensity_max', 'total_hw_days',
                                     'cumulative_intensity']

    if fmt == 'netcdf':
        # assemble a 2D (metric, lat, lon) variable set
        buffer = io.BytesIO()
        ds = __import__('netCDF4').Dataset('memory', mode='w',
                                           diskless=True,
                                           memory=buffer)
        ds.createDimension('metric', len(metrics))
        ds.createDimension('lat', grid.nlat)
        ds.createDimension('lon', grid.nlon)
        mv = ds.createVariable('metric', str, ('metric',))
        mv[:] = metrics
        latv = ds.createVariable('lat', 'f8', ('lat',))
        lonv = ds.createVariable('lon', 'f8', ('lon',))
        latv[:] = grid.lats
        lonv[:] = grid.lons
        for i, m in enumerate(metrics):
            arr = np.full((grid.nlat, grid.nlon), np.nan)
            for (i2, j2), v in result.metric_maps.get(m, {}).items():
                arr[i2, j2] = v
            vv = ds.createVariable(f'metric_{i}', 'f8', ('lat', 'lon'),
                                   zlib=True, fill_value=np.nan)
            vv.long_name = m
            vv[:] = arr
        ds.close()
        return send_file(buffer, mimetype='application/x-netcdf',
                         as_attachment=True,
                         download_name='spatial_heatwave_metrics.nc')

    # CSV / JSON: one row per cell
    rows = []
    for (i, j), v in result.metric_maps.get('n_events', {}).items():
        row = {'lat': round(float(grid.lats[i]), 5),
               'lon': round(float(grid.lons[j]), 5)}
        for m in metrics:
            row[m] = result.metric_maps.get(m, {}).get((i, j))
        # mask flag
        if APP_STATE.get('mask') is not None:
            row['in_study_area'] = bool(APP_STATE['mask'][i, j])
        else:
            row['in_study_area'] = True
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['lat', 'lon']).reset_index(drop=True)

    if fmt == 'json':
        return jsonify({'ok': True, 'rows': df.to_dict(orient='records'),
                        'columns': list(df.columns)})

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, mimetype='text/csv', as_attachment=True,
                     download_name='spatial_heatwave_metrics.csv')


@app.route('/api/export/report', methods=['GET'])
def export_report():
    """Export a JSON report of the last runs."""
    result = APP_STATE.get('spatial_result')
    grid = APP_STATE.get('grid')
    payload = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'grid': _grid_json_info(grid) if grid else None,
        'masked_cells': int(np.count_nonzero(APP_STATE['mask']))
            if APP_STATE.get('mask') is not None else None,
        'params': result.params if result else None,
        'summary': result.summarize() if result else None,
        'trends': APP_STATE.get('trends'),
    }
    buffer = io.BytesIO(json.dumps(_jsonable(payload), indent=2).encode('utf-8'))
    return send_file(buffer, mimetype='application/json', as_attachment=True,
                     download_name='spatial_heatwave_report.json')


@app.route('/api/reset', methods=['POST'])
def reset():
    APP_STATE['spatial_result'] = None
    APP_STATE['trends'] = None
    APP_STATE['year_stats'] = None
    APP_STATE['year_frames'] = None
    APP_STATE['secondary_grid'] = None
    APP_STATE['secondary_result'] = None
    return jsonify({'ok': True})


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, port=5030, host='127.0.0.1')
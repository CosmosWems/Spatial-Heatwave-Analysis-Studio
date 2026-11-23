"""End-to-end smoke test for the SPATIAL_HW application.

Creates synthetic gridded temperature data, runs the engine, and drives every
Flask endpoint through the test client to confirm the app responds correctly.
"""
import sys
import os
import io
import json
import zipfile
import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spatial_web_app as app_mod
import spatial_heatwave_engine as she

app = app_mod.app
app.config['TESTING'] = True

PASSED = []
FAILED = []


def check(name, cond, extra=''):
    if cond:
        PASSED.append(name)
        print(f'  [OK]   {name}')
    else:
        FAILED.append(name)
        print(f'  [FAIL] {name} {extra}')


def build_synthetic_netcdf(path, nyears=12, nlat=8, nlon=10):
    """Create a NetCDF file with a daily temperature field + injected events."""
    import netCDF4
    dates = pd.date_range('1991-01-01', periods=nyears * 365, freq='D')
    lats = np.linspace(4.5, 11.5, nlat)
    lons = np.linspace(-3.5, 1.5, nlon)

    rng = np.random.default_rng(7)
    doy = dates.dayofyear.to_numpy()
    years = dates.year.to_numpy()
    seasonal = 30 + 4 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    values = np.zeros((len(dates), nlat, nlon), dtype=np.float32)
    for i in range(nlat):
        for j in range(nlon):
            base = seasonal + 0.5 * (lats[i] - 8.0) + 0.2 * (lons[j] + 1.0)
            series = base + rng.normal(0, 1.0, len(dates))
            # inject heatwave events: +5C for 12-18 days, once or twice per
            # winter-spring season for the second half of the record
            cell_id = i * nlon + j
            for yr in range(years.min() + nyears // 2, years.max() + 1):
                for ev in range(2 if cell_id % 2 == 0 else 1):
                    doy_start = 20 + rng.integers(0, 200)
                    if doy_start + 18 > 365:
                        continue
                    span = (doy >= doy_start) & (doy < doy_start + 12 + cell_id % 7) & (years == yr)
                    series[span] += 5.0 + 0.5 * rng.normal()
            values[:, i, j] = series

    ds = netCDF4.Dataset(path, 'w', format='NETCDF4')
    ds.createDimension('time', len(dates))
    ds.createDimension('lat', nlat)
    ds.createDimension('lon', nlon)
    tvar = ds.createVariable('time', 'f8', ('time',))
    tvar[:] = netCDF4.date2num(dates.to_pydatetime(), 'days since 1991-01-01')
    tvar.units = 'days since 1991-01-01'
    latv = ds.createVariable('lat', 'f8', ('lat',))
    lonv = ds.createVariable('lon', 'f8', ('lon',))
    latv.units = 'degrees_north'
    lonv.units = 'degrees_east'
    latv[:] = lats
    lonv[:] = lons
    tv = ds.createVariable('t2m', 'f4', ('time', 'lat', 'lon'),
                           zlib=True, fill_value=np.float32(np.nan))
    tv.units = 'K'
    tv[:] = values
    tvmax = ds.createVariable('tasmax', 'f4', ('time', 'lat', 'lon'),
                              zlib=True, fill_value=np.float32(np.nan))
    tvmax.units = 'K'
    tvmax.standard_name = 'air_temperature'
    tvmax.long_name = 'Daily maximum near-surface air temperature'
    tvmax[:] = values + 3.0
    tvmin = ds.createVariable('tasmin', 'f4', ('time', 'lat', 'lon'),
                              zlib=True, fill_value=np.float32(np.nan))
    tvmin.units = 'K'
    tvmin.standard_name = 'air_temperature'
    tvmin.long_name = 'Daily minimum near-surface air temperature'
    tvmin[:] = values - 3.0
    ds.close()


def build_synthetic_shapefile_zip(zip_path):
    """Build a tiny polygon shapefile and zip it."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    gdf = gpd.GeoDataFrame(
        {'name': ['North', 'South']},
        geometry=[Polygon([(-3.5, 4.5), (1.5, 4.5), (1.5, 11.5), (-3.5, 11.5)]),
                  Polygon([(-3.5, 4.5), (1.5, 4.5), (1.5, 11.5), (-3.5, 11.5), (-3.5, 4.5)])]) \
        .set_crs(epsg=4326)
    tmp = zip_path + '_tmp'
    os.makedirs(tmp, exist_ok=True)
    gdf.to_file(os.path.join(tmp, 'boundary.shp'))
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(tmp):
            zf.write(os.path.join(tmp, fn), arcname=fn)
    return zip_path


def make_upload(file_like, filename, content_type='application/octet-stream'):
    return {'file': (file_like, filename, content_type)}


def main():
    print('=== SPATIAL_HW end-to-end test ===\n')
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_tmp')
    os.makedirs(tmp, exist_ok=True)

    nc_path = os.path.join(tmp, 'synthetic_t2m.nc')
    nc_zip_path = os.path.join(tmp, 'boundary.zip')

    build_synthetic_netcdf(nc_path)
    build_synthetic_shapefile_zip(nc_zip_path)
    print('Synthetic NetCDF + shapefile built.\n')

    # ---- 1. Health ----
    r = app.test_client().get('/api/health')
    data = r.get_json()
    check('health endpoint', data['status'] == 'ok' and data['modules']['netCDF4']
          and data['modules']['geopandas'], str(data['modules']))

    # ---- 2. Upload NetCDF ----
    with open(nc_path, 'rb') as f:
        r = app.test_client().post('/api/upload_data', data=make_upload(f, 'synthetic_t2m.nc'), content_type='multipart/form-data')
    data = r.get_json()
    check('upload netcdf', r.status_code == 200 and data['ok'] and data['info']['ntime'] > 100,
          str(data.get('error')))

    # ---- 2b. Variable discovery and TX/TN mapping ----
    r = app.test_client().get('/api/variables')
    data = r.get_json()
    var_names = [v['name'] for v in data.get('variables', [])]
    check('list variables', data['ok'] and len(var_names) >= 3 and
          'tasmax' in var_names and 'tasmin' in var_names, str(data.get('error')))

    r = app.test_client().post('/api/map_variables', json={
        'tmax': 'tasmax', 'tmin': 'tasmin', 'primary': 'tmax',
        'temperature_unit': 'auto'})
    data = r.get_json()
    check('map variables', r.status_code == 200 and data['ok'] and
          data['tx_info']['variable'] == 'tasmax' and
          data['tn_info']['variable'] == 'tasmin', str(data.get('error')))

    # ---- 3. Upload shapefile ----
    with open(nc_zip_path, 'rb') as f:
        r = app.test_client().post('/api/upload_shapefile', data=make_upload(f, 'boundary.zip'), content_type='multipart/form-data')
    data = r.get_json()
    check('upload shapefile', r.status_code == 200 and data['feature_count'] == 2,
          str(data.get('error')))
    r = app.test_client().post('/api/preprocess', json={
        'mask_with_shapefile': True, 'fill_gaps': True, 'smooth': False})
    data = r.get_json()
    check('preprocess mask', data['ok'] and data['masked_cells'] > 0,
          str(data.get('error')))

    grid = app_mod.APP_STATE['grid']
    mask = app_mod.APP_STATE['mask']
    check('mask covers domain', mask.shape == (grid.nlat, grid.nlon) and mask.sum() > 0)
    check('masked values are NaN outside', np.all(np.isnan(grid.values[:, ~mask])))

    # ---- 5. Detect (background job; poll /api/status) ----
    import time
    payload = {'clim_start': 1991, 'clim_end': 2001, 'pctile': 90, 'min_dur': 3,
               'join_gaps': False, 'max_gap': 2, 'smooth_pctile': True,
               'smooth_pctile_width': 31, 'window_half_width': 5, 'n_jobs': 2}

    def wait_done(client, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            st = client.get('/api/status').get_json()
            if st['phase'] == 'done':
                return st
            if st['phase'] == 'error':
                raise AssertionError('detect error: ' + str(st.get('error')))
            if st['phase'] == 'cancelled':
                raise AssertionError('detect cancelled unexpectedly')
            time.sleep(0.2)
        raise AssertionError('detect timed out')

    r = app.test_client().post('/api/detect', json=payload)
    data = r.get_json()
    check('detect started', r.status_code == 200 and data.get('started'),
          str(data.get('error')))
    st = wait_done(app.test_client())
    ls = st.get('last_summary') or {}
    check('detect completed ok', st['percent'] == 100.0 and ls.get('processed_cells', 0) > 0,
          str(st.get('label')))
    check('detect summary populated', 'n_events' in ls.get('summary', {}))

    res = app_mod.APP_STATE['spatial_result']
    check('grid per-cell results', any(res.mhws_map), 'no cell results')

    # ---- 6. Statistics ----
    r = app.test_client().post('/api/statistics', json={'year_start': 1991, 'year_end': 2020, 'block_length': 1})
    data = r.get_json()
    check('statistics', data['ok'] and len(data['regional_annual']) > 1,
          str(data.get('error')))

    # ---- 7. Trends (linear) ----
    r = app.test_client().post('/api/trends', json={'metric': 'n_events', 'method': 'linear',
                                                     'clim_start': 1991, 'clim_end': 2016})
    data = r.get_json()
    check('trends linear', data['ok'] and data['slope']['values'],
          str(data.get('error')))

    # ---- 8. Trends (mann-kendall) ----
    r = app.test_client().post('/api/trends', json={'metric': 'n_events', 'method': 'mannkendall',
                                                     'clim_start': 1991, 'clim_end': 2016})
    data = r.get_json()
    check('trends mann-kendall', data['ok'] and data['tau']['values'],
          str(data.get('error')))

    # ---- 9. Visualization: metric map ----
    r = app.test_client().post('/api/viz/metric_map', json={'metric': 'n_events'})
    data = r.get_json()
    fig = json.loads(data['plot']) if data['ok'] else None
    check('viz metric map', data['ok'] and fig and fig['data'],
          str(data.get('error')))
    check('plotly json is plain arrays (no bdata)',
          all('bdata' not in json.dumps(tr) for tr in fig['data']))
    # metric map must be a 2-D z field, not a flattened/stringified array
    z0 = fig['data'][0].get('z')
    check('metric map z is 2-D', isinstance(z0, list) and z0 and isinstance(z0[0], list),
          f'z type: {type(z0).__name__}')
    check('metric map has plain x (lons)',
          isinstance(fig['data'][0].get('x'), list)
          and isinstance(fig['data'][0]['x'][0], (int, float)))

    # ---- 9b. Data assessment field endpoint ----
    r = app.test_client().post('/api/preview/field', json={'mode': 'last'})
    data = r.get_json()
    field = data.get('field', {}) if data.get('ok') else {}
    check('preview field (last)', data['ok'] and len(field.get('values', [])) >=
          app_mod.APP_STATE['grid'].nlat, str(data.get('error')))
    r = app.test_client().post('/api/preview/field', json={'mode': 'mean'})
    data = r.get_json()
    field = data.get('field', {}) if data.get('ok') else {}
    check('preview field (mean)', data['ok'] and field.get('time_label') and
          'mean' in field['time_label'].lower(), str(data.get('error')))
    r = app.test_client().post('/api/preview/field', json={'mode': 'index', 'time_idx': 5})
    data = r.get_json()
    check('preview field (index)', data['ok'] and data['field']['time_idx'] == 5,
          str(data.get('error')))

    # ---- 10. Visualization: point series ----
    g = app_mod.APP_STATE['grid']
    lat0 = float(g.lats[g.nlat // 2]); lon0 = float(g.lons[g.nlon // 2])
    r = app.test_client().post('/api/viz/timeseries', json={'lat': lat0, 'lon': lon0})
    data = r.get_json()
    check('viz point series', data['ok'] and data['plot'],
          str(data.get('error')))
    fig = json.loads(data['plot'])
    tsv = next((tr for tr in fig['data'] if tr.get('name') == 'Temperature'), None)
    check('timeseries x are ISO date strings',
          tsv is not None and isinstance(tsv['x'][0], str)
          and tsv['x'][0][:4].isdigit() and '-' in tsv['x'][0],
          f'x0={tsv and tsv["x"][:1]}')
    check('timeseries y are plain numbers',
          tsv is not None and isinstance(tsv['y'][0], (int, float)))

    # ---- 11. Animation ----
    r = app.test_client().post('/api/viz/animation', json={'metric': 'count'})
    data = r.get_json()
    fig = json.loads(data['plot']) if data['ok'] else None
    check('viz animation', data['ok'] and len(fig.get('frames', [])) >= 2,
          str(data.get('error')))

    # ---- 12. Annual bars ----
    r = app.test_client().post('/api/viz/annual_bars', json={})
    data = r.get_json()
    check('annual bars', data['ok'] and data['plot'],
          str(data.get('error')))

    # ---- 13. Geojson boundaries endpoint ----
    r = app.test_client().get('/api/geojson_boundaries')
    data = r.get_json()
    check('geojson boundaries', len(data['boundaries']) > 0)

    # ---- 14. Comparison (period mode) ----
    r = app.test_client().post('/api/comparison/compare', json={
        'mode': 'period', 'clim_start_a': 1991, 'clim_end_a': 2000,
        'clim_start_b': 2001, 'clim_end_b': 2010, 'pctile': 90, 'min_dur': 3,
        'join_gaps': False})
    data = r.get_json()
    check('comparison period', data['ok'] and len(data['stats_table']) >= 5 and
          'n_events' in data['diff_maps'], str(data.get('error')))

    # ---- 15. Comparison (dataset mode) ----
    # upload secondary
    nc2 = os.path.join(tmp, 'synthetic_t2m_secondary.nc')
    build_synthetic_netcdf(nc2, nyears=15, nlat=8, nlon=10)
    with open(nc2, 'rb') as f:
        r = app.test_client().post('/api/comparison/upload_secondary',
                                   data=make_upload(f, 'secondary.nc'),
                                   content_type='multipart/form-data')
    data = r.get_json()
    check('upload secondary', data['ok'], str(data.get('error')))

    r = app.test_client().post('/api/comparison/compare', json={
        'mode': 'dataset', 'clim_start_a': 1991, 'clim_end_a': 2005,
        'pctile': 90, 'min_dur': 3, 'join_gaps': False})
    data = r.get_json()
    check('comparison dataset', data['ok'] and 'n_events' in data['diff_maps'],
          str(data.get('error')))

    # ---- 16. Export fields CSV ----
    r = app.test_client().post('/api/export/fields', json={'format': 'csv'})
    check('export csv', r.status_code == 200 and r.data.startswith(b'lat'),
          'csv header missing')

    # ---- 17. Export report ----
    r = app.test_client().get('/api/export/report')
    check('export report', r.status_code == 200 and json.loads(r.data)['generated_at'])

    # ---- 18. Data info ----
    r = app.test_client().get('/api/data_info')
    data = r.get_json()
    check('data info', data['loaded'] and data['mask_applied'])

    # ----------------------------------------------------------------
    # New endpoint checks: cores, status, stop, clear, error-handling,
    # exclude-sea preprocess, land mask upload, timeseries 400 JSON
    # ----------------------------------------------------------------

    # 19. Cores
    r = app.test_client().get('/api/cores')
    data = r.get_json()
    check('cores endpoint', r.status_code == 200 and data.get('total', 0) >= 1,
          str(data.get('error')))

    # 20. Status
    r = app.test_client().get('/api/status')
    data = r.get_json()
    check('status endpoint', r.status_code == 200 and data['phase'] == 'done')

    # 21. Stop when idle
    r = app.test_client().post('/api/stop')
    data = r.get_json()
    check('stop when idle', r.status_code == 200 and not data.get('cancelling'))

    # 22. timeseries after detect works
    r = app.test_client().post('/api/viz/timeseries',
                               json={'lat': float(grid.lats[0]), 'lon': float(grid.lons[0])})
    data = r.get_json()
    check('timeseries after detect ok', r.status_code == 200 and data.get('ok'),
          f'err={data.get("error")}')

    # 23. Clear
    r = app.test_client().post('/api/clear')
    data = r.get_json()
    check('clear endpoint', r.status_code == 200 and data.get('ok'))

    # 24. Grid cleared
    r = app.test_client().get('/api/data_info')
    data = r.get_json()
    check('cleared grid info', r.status_code == 200 and not data.get('loaded'))

    # 25. timeseries after clear returns JSON 400 (not HTML)
    r = app.test_client().post('/api/viz/timeseries', json={'lat': 0, 'lon': 0})
    data = r.get_json()
    check('timeseries after clear returns JSON 400',
          r.status_code == 400 and isinstance(data, dict) and 'error' in data,
          f'status={r.status_code}')

    # 26. Preprocess with exclude_sea (re-upload netcdf first)
    with open(nc_path, 'rb') as f:
        r = app.test_client().post('/api/upload_data',
                                   data=make_upload(f, 'temperature.nc'),
                                   content_type='multipart/form-data')
    data = r.get_json()
    check('re-upload after clear', data.get('ok'), str(data.get('error')))

    r = app.test_client().post('/api/preprocess', json={
        'mask_with_shapefile': False, 'exclude_sea': True, 'fill_gaps': True})
    data = r.get_json()
    check('preprocess with exclude_sea',
          data['ok'] and data.get('excluded_sea') and 'elapsed_ms' in data,
          f'status-error={data.get("error")}')

    # 27. Upload static land mask (2-D NetCDF)
    import netCDF4 as _nc4
    mask_nc_path = os.path.join(os.path.dirname(nc_path), 'land_mask.nc')
    ds_mask = _nc4.Dataset(mask_nc_path, 'w')
    ds_mask.createDimension('lat', 8)
    ds_mask.createDimension('lon', 10)
    ds_mask.createVariable('lat', 'f8', ('lat',))[:] = np.linspace(4.5, 11.5, 8)
    ds_mask.createVariable('lon', 'f8', ('lon',))[:] = np.linspace(-3.5, 1.5, 10)
    mask_var = ds_mask.createVariable('land', 'f4', ('lat', 'lon'))
    mask_var[:] = np.ones((8, 10), dtype=np.float32)
    ds_mask.close()
    with open(mask_nc_path, 'rb') as f:
        r = app.test_client().post('/api/upload_landmask',
                                   data=make_upload(f, 'land.nc'),
                                   content_type='multipart/form-data')
    data = r.get_json()
    check('upload landmask', r.status_code == 200 and data.get('ok'),
          str(data.get('error')))

    print('\n=== RESULTS ===')
    print(f'PASSED: {len(PASSED)}, FAILED: {len(FAILED)}')
    if FAILED:
        print('Failed: ' + ', '.join(FAILED))
        sys.exit(1)


if __name__ == '__main__':
    main()
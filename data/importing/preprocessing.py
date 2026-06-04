import xarray as xr
import numpy as np
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta

base = Path("/nobackup/users/haar/radar_data/MFBS/1H20-23")
out_base = Path("/nobackup/users/haar/radar_data/MFBS/6H20-23-preprocessed")
prefix_in = "RAD_NL25_RAC_MFBS_01H_"
prefix_out = "RAD_NL25_RAC_MFBS_06H_"
var_in = "image1_image_data"
var_out = "rgar"

def compute_latlon(filepath):
    ds_nc = nc.Dataset(filepath)
    corners = ds_nc.variables['geographic'].geo_product_corners
    ds_nc.close()
    bl_lon, bl_lat = corners[0], corners[1]
    tl_lon, tl_lat = corners[2], corners[3]
    tr_lon, tr_lat = corners[4], corners[5]
    br_lon, br_lat = corners[6], corners[7]
    row_frac = np.linspace(0, 1, 765)
    col_frac = np.linspace(0, 1, 700)
    cc, rr = np.meshgrid(col_frac, row_frac)
    lats = (tl_lat * (1-rr) * (1-cc) +
            tr_lat * (1-rr) * cc +
            bl_lat * rr * (1-cc) +
            br_lat * rr * cc)
    lons = (tl_lon * (1-rr) * (1-cc) +
            tr_lon * (1-rr) * cc +
            bl_lon * rr * (1-cc) +
            br_lon * rr * cc)
    return lats, lons

errors = []
for year in range(2020, 2024):
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                datetime(year, month, day)
            except ValueError:
                continue
            for window_start_hour in [0, 6, 12, 18]:
                window_start = datetime(year, month, day, window_start_hour)
                files = [
                    base / t.strftime('%Y/%m') / f"{prefix_in}{t.strftime('%Y%m%d%H%M')}.nc"
                    for t in (window_start + timedelta(hours=h) for h in range(6))
                ]
                try:
                    datasets = [xr.open_dataset(f) for f in files]
                    accumulated = sum(ds[var_in].isel(time=0) for ds in datasets)
                    out_ds = accumulated.to_dataset(name=var_out)
                    out_ds[var_out].attrs = datasets[0][var_in].attrs
                    out_ds.attrs = datasets[0].attrs
                    for var_name in ['projection', 'geographic', 'product']:
                        if var_name in datasets[0]:
                            out_ds[var_name] = datasets[0][var_name]
                    out_ds = out_ds.assign_coords(time=window_start)
                    lats, lons = compute_latlon(files[0])
                    out_ds['latitude'] = xr.DataArray(
                        lats, dims=['y', 'x'],
                        attrs={'units': 'degrees_north', 'long_name': 'latitude', 'standard_name': 'latitude'}
                    )
                    out_ds['longitude'] = xr.DataArray(
                        lons, dims=['y', 'x'],
                        attrs={'units': 'degrees_east', 'long_name': 'longitude', 'standard_name': 'longitude'}
                    )
                    out_ds[var_out].attrs['coordinates'] = 'latitude longitude'
                    out_ds[var_out].attrs['grid_mapping'] = 'projection'
                    out_path = out_base / window_start.strftime('%Y/%m')
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / f"{prefix_out}{window_start.strftime('%Y%m%d%H%M')}.nc"
                    out_ds.to_netcdf(out_file)
                    for ds in datasets:
                        ds.close()
                    print(f"Written: {out_file.name}")
                except Exception as e:
                    print(f"Error at {window_start}: {e}")
                    errors.append(window_start)

print(f"\nDone. {len(errors)} windows had errors or missing files:")
for e in errors:
    print(f"  {e}")
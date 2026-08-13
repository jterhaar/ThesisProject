import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from pyproj import CRS, Transformer

base = Path("/nobackup/users/haar/radar_data/MFBS/1H20-23")
out_base = Path("/nobackup/users/haar/radar_data/MFBS/6H20-23-reprojected")

prefix_in = "RAD_NL25_RAC_MFBS_01H_"
prefix_out = "RAD_NL25_RAC_MFBS_06H_"

var_in = "image1_image_data"
var_out = "tp"

errors = []

# -----------------------------
# lat/lon cache
# -----------------------------
latlon_cache = {}

def get_latlon(ds):
    key = (ds.sizes["y"], ds.sizes["x"])

    if key in latlon_cache:
        return latlon_cache[key]

    crs = CRS.from_cf(ds["projection"].attrs)
    transformer = Transformer.from_crs(crs, 4326, always_xy=True)

    x_m = ds["x"].values * 1000.0
    y_m = ds["y"].values * 1000.0

    X, Y = np.meshgrid(x_m, y_m)

    longitude, latitude = transformer.transform(X, Y)

    latlon_cache[key] = (latitude, longitude)
    return latitude, longitude


# -----------------------------
# TEST RANGE: first week Jan 2020
# -----------------------------
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

current = start_date

while current <= end_date:

    # window ends are 06:00, 12:00, 18:00, and 00:00 of the *next* day
    for window_end_hour in [6, 12, 18, 24]:

        window_end = current + timedelta(hours=window_end_hour)

        # the 6 one-hour files whose START times fall in
        # [window_end - 6h, window_end - 1h], i.e. the 6 hours BEFORE window_end
        files = [
            base / t.strftime('%Y/%m') /
            f"{prefix_in}{t.strftime('%Y%m%d%H%M')}.nc"
            for t in (window_end - timedelta(hours=h) for h in range(6, 0, -1))
        ]

        try:
            accum = None
            ds0 = None

            for i, f in enumerate(files):

                if not f.exists():
                    raise FileNotFoundError(f)

                with xr.open_dataset(f) as ds:

                    if ds0 is None:
                        ds0 = ds

                    data = ds[var_in].isel(time=0).load()

                    if accum is None:
                        accum = data
                    else:
                        accum = accum + data

            with xr.open_dataset(files[0]) as ds_ref:
                latitude, longitude = get_latlon(ds_ref)

                out = accum.to_dataset(name=var_out)

                out[var_out].attrs = ds_ref[var_in].attrs
                out.attrs = ds_ref.attrs

                for v in ["projection", "geographic", "product"]:
                    if v in ds_ref:
                        out[v] = ds_ref[v]

                out = out.assign_coords(
                    time=window_end,
                    latitude=(("y", "x"), latitude),
                    longitude=(("y", "x"), longitude),
                )

            out_path = out_base / window_end.strftime('%Y/%m')
            out_path.mkdir(parents=True, exist_ok=True)

            out_file = out_path / (
                f"{prefix_out}{window_end.strftime('%Y%m%d%H%M')}.nc"
            )

            out.to_netcdf(out_file)

            print(f"Written: {out_file.name}")

        except Exception as e:
            print(f"Error at {window_end}: {e}")
            errors.append(window_end)

    current += timedelta(days=1)


print(f"\nDone. {len(errors)} windows had errors or missing files:")
for e in errors:
    print(f"  {e}")
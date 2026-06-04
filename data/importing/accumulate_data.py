import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

base = Path("/nobackup/users/haar/radar_data/MFBS/1H20-23")
out_base = Path("/nobackup/users/haar/radar_data/MFBS/6H20-23")
prefix_in = "RAD_NL25_RAC_MFBS_01H_"
prefix_out = "RAD_NL25_RAC_MFBS_06H_"
var_in = "image1_image_data"
var_out = "rgar"

errors = []

for year in range(2020, 2024):
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                datetime(year, month, day)
            except ValueError:
                continue  # skip invalid dates e.g. Feb 30

            for window_start_hour in [0, 6, 12, 18]:
                window_start = datetime(year, month, day, window_start_hour)

                # Collect 6 files: window_start+0h to window_start+5h
                files = [
                    base / t.strftime('%Y/%m') / f"{prefix_in}{t.strftime('%Y%m%d%H%M')}.nc"
                    for t in (window_start + timedelta(hours=h) for h in range(6))
                ]

                try:
                    # Sum the 6 hourly fields
                    datasets = [xr.open_dataset(f) for f in files]
                    accumulated = sum(ds[var_in].isel(time=0) for ds in datasets)

                    # Wrap back into a dataset with the correct time coordinate
                    out_ds = accumulated.to_dataset(name=var_out)
                    out_ds[var_out].attrs = datasets[0][var_in].attrs
                    out_ds.attrs = datasets[0].attrs

                    # Copy variables
                    for var_name in ['projection', 'geographic', 'product']:
                        if var_name in datasets[0]:
                            out_ds[var_name] = datasets[0][var_name]

                    # Assign time coordinate to window start
                    out_ds = out_ds.assign_coords(time=window_start)

                    # Write output
                    out_path = out_base / window_start.strftime('%Y/%m')
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_file = out_path / f"{prefix_out}{window_start.strftime('%Y%m%d%H%M')}.nc"
                    out_ds.to_netcdf(out_file)

                    # Close datasets to avoid file handle leaks
                    for ds in datasets:
                        ds.close()

                    print(f"Written: {out_file.name}")

                except Exception as e:
                    print(f"Error at {window_start}: {e}")
                    errors.append(window_start)

print(f"\nDone. {len(errors)} windows had errors or missing files:")
for e in errors:
    print(f"  {e}")
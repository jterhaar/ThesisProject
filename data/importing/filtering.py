import netCDF4 as nc
import numpy as np
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load removal mask
flat_remove = np.load('data/analysis/flat_remove.npy')
remove_grid = np.zeros((765, 700), dtype=bool)
ys, xs = np.unravel_index(flat_remove, (765, 700))
remove_grid[ys, xs] = True

start = datetime(2020, 1, 1, 0, 0)
end   = datetime(2023, 12, 31, 18, 0)
dates = []
d = start
while d <= end:
    dates.append(d)
    d += timedelta(hours=6)

src_base = Path("/nobackup/users/haar/radar_data/MFBS/6H20-23-reprojected")
dst_base = Path("/nobackup/users/haar/radar_data/MFBS/6H20-23-final")

# Pre-create output directories
for m in {d.strftime('%Y/%m') for d in dates}:
    (dst_base / m).mkdir(parents=True, exist_ok=True)

def process(args):
    date_str, src_str, dst_str, ys, xs = args
    src = Path(src_str)
    dst = Path(dst_str)
    remove = np.zeros((765, 700), dtype=bool)
    remove[ys, xs] = True
    shutil.copy2(src, dst)
    with nc.Dataset(dst, 'r+') as ds:
        tp = ds['tp'][:]
        tp[remove] = np.nan
        ds['tp'][:] = tp

# Build args list — pass ys/xs instead of remove_grid (can't pickle bool array easily across processes)
args_list = [
    (
        d.strftime('%Y%m%d%H%M'),
        str(src_base / d.strftime('%Y/%m') / f"RAD_NL25_RAC_MFBS_06H_{d.strftime('%Y%m%d%H%M')}.nc"),
        str(dst_base / d.strftime('%Y/%m') / f"RAD_NL25_RAC_MFBS_06H_{d.strftime('%Y%m%d%H%M')}.nc"),
        ys, xs,
    )
    for d in dates
]

with ProcessPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(process, a): a for a in args_list}
    with tqdm(total=len(dates), unit='file', dynamic_ncols=True) as pbar:
        for fut in as_completed(futures):
            fut.result()
            pbar.update(1)
from pathlib import Path
import xarray as xr

prefix = 'RAD_NL25_RAC_MFBS_01H_'

#Files have 1-based hour numbering, we decrease it by one in the filename for easy indexing with pattern. 
for year in range(2020, 2024):
    for month in range(1, 13):
        dirpath = f"/nobackup/users/haar/radar_data/MFBS/NetCDF2020-23/{year}/{month:02d}"
        for f in sorted(Path(dirpath).iterdir()):
            filename = f.name

            hour = int(filename[len(prefix)+8:len(prefix)+10])
            hour -= 1
            new_name = f"{filename[:len(prefix)+8]}{hour:02d}00.nc"
            f.rename(f.parent / new_name)
 

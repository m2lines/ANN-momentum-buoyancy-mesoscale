"""Stage 1 of the rho-gradient pilot: pull CM2.6 raw temp/salt from the cloud to OUR scratch.

Pavel's raw (/scratch/pp2681/.../rawdata) has rho/u/v but DROPPED temp/salt at download
(download_raw_data.py:18). We pull just the missing T/S (full resolution) once, so coarse-
graining to ANY factor reads from local disk instead of re-streaming the cloud each time.

Mirrors download_raw_data.py exactly -- same source (cmip6-3d), same train/validate/test time
selectors, same DatasetCM26 path -- so the saved T/S align with Pavel's rho/u/v grid. We only
need temp/salt (set on the h-grid in from_cloud, no velocity interp), so compute_param=False
skips the expensive global mask compute; the fine param already lives in Pavel's param.nc.

Writes {split}-{j}.nc (temp, salt) to CM26_RAWDATA (default our scratch rawdata dir).

  SPLIT train|validate|test    START/END snapshot range  [END=0 -> to end of split]
"""
import os
import sys
import cftime
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import DatasetCM26

SPLIT = os.environ.get('SPLIT', 'test')
START = int(os.environ.get('START', 0))
END = int(os.environ.get('END', 0))                 # 0 = to the end of the split
OUT = os.path.expandvars(os.environ.get('CM26_RAWDATA', '/scratch/$USER/CM26_datasets/ocean3d/rawdata'))
os.makedirs(OUT, exist_ok=True)

# the exact snapshots each split selects (identical to download_raw_data.py)
YEARS = {'train': range(181, 189), 'validate': range(194, 195), 'test': range(199, 201)}
dates = [cftime.DatetimeJulian(y, m, 15) for y in YEARS[SPLIT] for m in range(1, 13)]
end = END if END else len(dates)
print(f'split {SPLIT}: snaps [{START},{end}) of {len(dates)} -> {OUT}', flush=True)

print('opening cloud cmip6-3d (compute_param=False) ...', flush=True)
fine = DatasetCM26(source='cmip6-3d', compute_param=False)
ts = fine.data[['temp', 'salt']].sel(time=dates, method='nearest')

for t in range(START, end):
    outfile = f'{OUT}/{SPLIT}-{t}.nc'
    if os.path.exists(outfile):
        print(f'  [{t}] exists, skip', flush=True)
        continue
    ts.isel(time=t).to_netcdf(outfile)
    sz = os.path.getsize(outfile) / 1e9
    print(f'  [{t}] wrote {outfile}  ({sz:.1f} GB)', flush=True)

print('done', flush=True)

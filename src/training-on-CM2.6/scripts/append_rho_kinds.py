"""Append density-gradient "kind" variants to factor-N coarse data (pilot; see ../PLAN.md).

Online, MOM6's calc_isoneutral_slopes builds the ANN's density-gradient input from the EOS
derivatives at the LOCAL interface pressure (a neutral / locally-referenced gradient). Offline,
training feeds the gradient of surface-referenced sigma0. To test whether that mismatch changes
the trained model, we coarse-grain CM2.6 temp/salt -- which exist ONLY in the cloud (the local
raw dropped them at download) -- with the SAME Filtering()+CoarsenKochkovMinMax() operators the
production rho uses, then build two horizontal density gradients from the coarse T/S:

  rhox_sigma0c / rhoy_sigma0c : rho = gsw.rho(S,T, p=0)     (offline reference, sigma0 from coarse T/S)
  rhox_neutral / rhoy_neutral : rho = gsw.rho(S,T, p=zl)    (online analog, local interface pressure)

Both come from the SAME coarse T/S, so they differ ONLY by reference pressure. The existing
production rhox/rhoy (= gradient of coarse-grained full-res sigma0) is kept as the third variant.
The sigma0 flux target Fx/Fy is unchanged (paper buoyancy definition). We write one sidecar file
per snapshot with temp, salt + the four gradient fields, on the existing coarse grid.

  FACTOR 9  FGR 3  PERCENTILE 0.5   (must match the existing factor-N filter.txt)
  SPLIT train|validate|test   START/END snapshot range   ZLMAX (smoke: only top levels)
  OUT  output dir   [default scratch .../factor-{FACTOR}-rhokinds]
"""
import os
import sys
import types
import cftime
import numpy as np
import xarray as xr
import gsw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import DatasetCM26, create_grid
from helpers.operators import Filtering, CoarsenKochkovMinMax

FACTOR = int(os.environ.get('FACTOR', 9))
FGR = int(os.environ.get('FGR', 3))
PERCENTILE = float(os.environ.get('PERCENTILE', 0.5))
SPLIT = os.environ.get('SPLIT', 'test')
START = int(os.environ.get('START', 0))
END = int(os.environ.get('END', 0))          # 0 = to the end of the split
ZLMAX = int(os.environ.get('ZLMAX', 0))      # 0 = all 50 levels; else only top ZLMAX (smoke)
SOURCE = os.environ.get('SOURCE', 'local')   # 'local' = our raw T/S on disk; 'cloud' = stream cmip6-3d
RHO0 = 1025.0                                 # land BC fill, matches StateFunctions.rho()
ROOT = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d/subfilter')
OUT = os.path.expandvars(os.environ.get('OUT', f'{ROOT}/FGR{FGR}/factor-{FACTOR}-rhokinds'))
EXISTING = f'{ROOT}/FGR{FGR}/factor-{FACTOR}'
RAWTS = os.path.expandvars(os.environ.get('CM26_RAWDATA', '/scratch/$USER/CM26_datasets/ocean3d/rawdata'))
FINE_PARAM = os.environ.get('FINE_PARAM', '/scratch/pp2681/CM26_datasets/ocean3d/rawdata/param.nc')
os.makedirs(OUT, exist_ok=True)

# the exact snapshots each split selects (replicates download_raw_data.py)
YEARS = {'train': range(181, 189), 'validate': range(194, 195), 'test': range(199, 201)}
dates = [cftime.DatetimeJulian(y, m, 15) for y in YEARS[SPLIT] for m in range(1, 13)]
end = END if END else len(dates)
print(f'factor {FACTOR} FGR{FGR} pct{PERCENTILE} | split {SPLIT} snaps [{START},{end}) of {len(dates)}'
      f' | ZLMAX={ZLMAX or 50}', flush=True)

# coarse grid: load the EXISTING factor-N param (deterministic -> aligns with the coarse files
# we are appending to), then its xgcm grid. Avoids recomputing init_coarse_grid.
cparam = xr.open_dataset(f'{EXISTING}/param.nc')
if ZLMAX:
    cparam = cparam.isel(zl=slice(0, ZLMAX))
cgrid = create_grid(cparam)

# fine temp/salt source. 'local' reads our pre-pulled raw + Pavel's fine param (no cloud, no
# global mask compute); 'cloud' streams cmip6-3d (fallback if the local raw is gone).
if SOURCE == 'cloud':
    print('opening cloud cmip6-3d ...', flush=True)
    fine = DatasetCM26(source='cmip6-3d')
    fdata = fine.data.sel(time=dates, method='nearest')
    fparam = fine.param
    def get_TS(t):
        return fdata.temp.isel(time=t), fdata.salt.isel(time=t)
else:
    print(f'reading local raw T/S from {RAWTS}, fine param {FINE_PARAM}', flush=True)
    fparam = xr.open_dataset(FINE_PARAM)
    def get_TS(t):
        d = xr.open_dataset(f'{RAWTS}/{SPLIT}-{t}.nc', chunks={'zl': 1})
        return d.temp, d.salt
if ZLMAX and 'zl' in fparam.dims:
    fparam = fparam.isel(zl=slice(0, ZLMAX))
FGR_ABS = FACTOR * FGR                                    # absolute filter scale on the fine grid
filt = Filtering()                                        # Gaussian, matches filter.txt
coarsen = CoarsenKochkovMinMax()
fine_h = types.SimpleNamespace(param=fparam)              # operators only touch .param
coarse_h = types.SimpleNamespace(param=cparam)


def coarse_grain(field):
    """filter on the fine grid (FGR_ABS) then block-coarsen to the coarse grid -- the scalar path,
    identical to how production coarse-grains rho."""
    _, _, f = filt(None, None, field, fine_h, FGR_ABS)
    _, _, c = coarsen(None, None, f, fine_h, coarse_h, factor=FACTOR)
    return c.load()


def rho_grad(rho):
    """horizontal density gradient on the coarse grid, replicating vertical_shear_geostrophic."""
    rho = cparam.wet * rho + (1 - cparam.wet) * RHO0     # land BC fill (as StateFunctions.rho)
    r = rho.chunk({'zl': -1})
    rhoy = cgrid.interp(cgrid.diff(r, 'Y') / cparam.dyCv * cparam.wet_v, 'Y') * cparam.wet
    rhox = cgrid.interp(cgrid.diff(r, 'X') / cparam.dxCu * cparam.wet_u, 'X') * cparam.wet
    return rhox, rhoy


for t in range(START, end):
    outfile = f'{OUT}/{SPLIT}-{t}.nc'
    if os.path.exists(outfile):
        print(f'  [{t}] exists, skip', flush=True)
        continue
    temp, salt = get_TS(t)
    tcoord = temp['time']
    if ZLMAX:
        temp, salt = temp.isel(zl=slice(0, ZLMAX)), salt.isel(zl=slice(0, ZLMAX))
    temp_c = coarse_grain(temp)
    salt_c = coarse_grain(salt)
    # stamp the canonical coarse coords so the sidecar aligns with the existing files
    temp_c = temp_c.assign_coords(zl=cparam.zl, yh=cparam.yh, xh=cparam.xh)
    salt_c = salt_c.assign_coords(zl=cparam.zl, yh=cparam.yh, xh=cparam.xh)

    rho_s0 = gsw.rho(salt_c, temp_c, 0.0)                 # sigma0-equivalent (= 1000+sigma0)
    rho_nt = gsw.rho(salt_c, temp_c, cparam.zl)           # local interface pressure ~ depth in dbar
    rhox_s0, rhoy_s0 = rho_grad(rho_s0)
    rhox_nt, rhoy_nt = rho_grad(rho_nt)

    out = xr.Dataset({
        'temp': temp_c, 'salt': salt_c,
        'rhox_sigma0c': rhox_s0, 'rhoy_sigma0c': rhoy_s0,
        'rhox_neutral': rhox_nt, 'rhoy_neutral': rhoy_nt,
    }).astype('float32')
    out = out.assign_coords(time=tcoord)
    out.to_netcdf(outfile)
    print(f'  [{t}] wrote {outfile}  rho_s0 mean={float(rho_s0.where(cparam.wet==1).mean()):.2f}', flush=True)

print('done', flush=True)

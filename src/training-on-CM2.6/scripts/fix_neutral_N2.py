"""Replace the compression-contaminated N_buoyancy in the neutral dataset with the correct
locally-referenced (neutral) buoyancy frequency (rho-gradient pilot; see ../PLAN.md).

The pipeline's Nsquared diffs in-situ rho across levels with VARYING reference pressure
(rho = gsw.rho(S,T,p=zl)), so for the neutral set d(rho)/dz picks up the adiabatic compression
(~several kg/m^3 per km) and N_buoyancy is dominated by it, not stratification. We recompute the
correct N the locally-referenced (MOM6) way -- compression cancels because both levels of each
interface use the same local pressure:

    N^2 = g * ( -alpha(p) * dtheta/dz + beta(p) * dS/dz )   (alpha,beta = gsw at interface p)
    N   = sqrt(max(N^2, 0)),  then interp interface->center  (matches state_functions:2214)

Coarse T/S come from factor-{F}-rhokinds (same coarse grid + snapshots). We overwrite N_buoyancy
in subfilter-neutral so each set's N_buoyancy is the N^2 consistent with its density (sigma0 set
keeps d(sigma0)/dz, which is correct). TEST=1 -> just compare test-0 vs the sigma0 N_buoyancy.

  FACTOR 9   SPLIT (TEST mode ignores)   TEST 0|1
"""
import sys
import os
import numpy as np
import xarray as xr
import gsw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import create_grid

FACTOR = int(os.environ.get('FACTOR', 9))
FGR = int(os.environ.get('FGR', 3))
TEST = int(os.environ.get('TEST', 1))
G, RHO0 = 9.8, 1025.0
ROOT = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d')
NEU = f'{ROOT}/subfilter-neutral/FGR{FGR}/factor-{FACTOR}'
SIG = f'{ROOT}/subfilter-sigma0/FGR{FGR}/factor-{FACTOR}'
RK = f'{ROOT}/subfilter/FGR{FGR}/factor-{FACTOR}-rhokinds'         # coarse T/S live here
param = xr.open_dataset(f'{NEU}/param.nc')
grid = create_grid(param)
dzB = grid.diff(param.zl, 'Z')                                     # interface spacing
pi = grid.interp(param.zl.astype('float64'), 'Z')                 # interface pressure ~ depth (dbar)


def neutral_N(temp, salt):
    """correct locally-referenced buoyancy frequency N on cell centers (float64)."""
    t = temp.astype('float64').chunk({'zl': -1})
    s = salt.astype('float64').chunk({'zl': -1})
    ti, si = grid.interp(t, 'Z'), grid.interp(s, 'Z')             # T,S at interfaces
    alpha = gsw.alpha(si, ti, pi)                                 # thermal expansion at local p
    beta = gsw.beta(si, ti, pi)                                   # haline contraction at local p
    dtdz = grid.diff(t, 'Z') / dzB                                # Z is downward (matches Nsquared)
    dsdz = grid.diff(s, 'Z') / dzB
    N2 = np.maximum(G * (-alpha * dtdz + beta * dsdz), 0.0) * param.wet_w
    return grid.interp(np.sqrt(N2), 'Z') * param.wet              # interface -> center, like the pipeline


if TEST:
    rk = xr.open_dataset(f'{RK}/test-0.nc')
    sig = xr.open_dataset(f'{SIG}/test-0.nc')
    neu = xr.open_dataset(f'{NEU}/test-0.nc')
    Nn = neutral_N(rk.temp, rk.salt).compute()
    wet = param.wet.values > 0.5
    zl = param.zl.values
    print('locally-referenced neutral N vs sigma0 N_buoyancy vs contaminated neutral N_buoyancy:')
    print('  depth     N_neutral   N_sigma0   N_contam   (median over wet, 1/s)')
    for lo, hi in [(0, 200), (200, 1000), (1000, 2500), (2500, 5500)]:
        m = (zl >= lo)[:, None, None] & (zl < hi)[:, None, None] & wet
        a = Nn.values[m]; b = sig.N_buoyancy.values[m]; c = neu.N_buoyancy.values[m]
        f = np.isfinite(a) & np.isfinite(b)
        print(f'  {lo:4d}-{hi:4d}m  {np.nanmedian(a[f]):.3e}  {np.nanmedian(b[f]):.3e}  {np.nanmedian(c[f]):.3e}')
    print('\nsanity: N_neutral finite frac=%.4f, negatives=%d' %
          (np.isfinite(Nn.values[wet]).mean(), int((Nn.values[wet] < 0).sum())))
else:
    for split, nf in [('train', 96), ('validate', 12), ('test', 24)]:
        for t in range(nf):
            f = f'{NEU}/{split}-{t}.nc'
            rk = xr.open_dataset(f'{RK}/{split}-{t}.nc')
            Nn = neutral_N(rk.temp, rk.salt).compute().astype('float32')
            d = xr.open_dataset(f).load()
            d['N_buoyancy'] = d['N_buoyancy'].copy(data=Nn.transpose(*d['N_buoyancy'].dims).values)
            d.to_netcdf(f + '.tmp'); os.replace(f + '.tmp', f)
            print(f'{split} [{t}] N_buoyancy -> locally-referenced', flush=True)
    print('DONE', flush=True)

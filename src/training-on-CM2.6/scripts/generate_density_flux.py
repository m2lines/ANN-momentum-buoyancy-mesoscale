"""Generate a factor-N coarse dataset with a chosen DENSITY reference (rho-gradient pilot).

Builds two directly-comparable sets (identical pipeline, only the density differs) so we can
compare sigma0 vs neutral for the flux target, gradient, N^2, and every density-derived field:

  DENSITY=sigma0   rho = gsw.rho(S,T, p=0)      -> surface-referenced potential density
  DENSITY=neutral  rho = gsw.rho(S,T, p=local)  -> locally-referenced (neutral), GM-principled

We reuse the PROVEN fine-dataset loading (the '3d-' source: reads Pavel's u/v/rho raw and does
param.compute().chunk(), which the original generation relies on -- a lazy/unchunked 10.8 GB
param OOMs even at 128 GB), then OVERWRITE rho with the chosen density computed from our pulled
T/S, and run the UNCHANGED compute_subfilter_forcing. So the flux target (Fx,Fy), gradient
(rhox,rhoy), N^2 (N_buoyancy), deformation radius, etc. all come out consistent with the density.
rho is cast to float32 to match the original pipeline (gsw returns float64 -> 2x memory).

Output -> subfilter-{DENSITY}/FGR{FGR}/factor-{FACTOR}, trainable via
read_datasets(subfilter='subfilter-{DENSITY}').

  DENSITY sigma0|neutral   SPLIT train|validate|test   NSNAP (0=all; >0 first N, smoke)
"""
import sys
import os
import numpy as np
import xarray as xr
import gsw

DENSITY = os.environ.get('DENSITY', 'neutral')
assert DENSITY in ('sigma0', 'neutral'), f'DENSITY must be sigma0|neutral, got {DENSITY}'
FACTOR = int(os.environ.get('FACTOR', 9))
FGR = int(os.environ.get('FGR', 3))
PERCENTILE = float(os.environ.get('PERCENTILE', 0.5))
SPLIT = os.environ.get('SPLIT', 'test')
NSNAP = int(os.environ.get('NSNAP', 0))                  # 0 = all; >0 = first N (smoke)
RHO0 = 1025.0
UV = os.environ.get('UV_RAW', '/scratch/pp2681/CM26_datasets/ocean3d/rawdata')          # Pavel: u,v,rho,param
TS = os.path.expandvars(os.environ.get('TS_RAW', '/scratch/$USER/CM26_datasets/ocean3d/rawdata'))  # ours: T,S
ROOT = os.path.expandvars(os.environ.get('CM26_DATA_ROOT', '/scratch/$USER/CM26_datasets/ocean3d'))
OUT = f'{ROOT}/subfilter-{DENSITY}/FGR{FGR}/factor-{FACTOR}'
os.makedirs(OUT, exist_ok=True)
NF = {'train': 96, 'validate': 12, 'test': 24}[SPLIT]
depth_selector = lambda x: x.isel(zl=np.arange(0, 50, 1))

# the '3d-' source reads u/v/rho + param from CM26_RAWDATA; point it at Pavel's raw.
os.environ['CM26_RAWDATA'] = UV
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import DatasetCM26
from helpers.operators import CoarsenKochkovMinMax, Filtering

# proven loading: chunked u/v/rho + param.compute().chunk()
ds = DatasetCM26(source=f'3d-{SPLIT}')
if NSNAP:
    ds.data = ds.data.isel(time=slice(0, NSNAP))

# overwrite rho with the chosen density, from our T/S (same grid + snapshots as Pavel's raw)
ts = xr.open_mfdataset([f'{TS}/{SPLIT}-{j}.nc' for j in range(NF)],
                       chunks={'zl': 1, 'time': 1}, concat_dim='time', combine='nested')
if NSNAP:
    ts = ts.isel(time=slice(0, NSNAP))
ts = ts.assign_coords(time=ds.data.time, zl=ds.data.zl, yh=ds.data.yh, xh=ds.data.xh)
p_ref = 0.0 if DENSITY == 'sigma0' else ds.param.zl            # local pressure (zl ~ dbar) for neutral
# Keep density in float64 (gsw's native output). rho ~ 1025, and both the gradient (d rho) and the
# flux (rho_bar*u_bar - (rho*u)_bar) are SMALL differences of large numbers -> catastrophic
# cancellation in float32 (~1.2e-4 resolution at 1025 => up to ~10% noise on weak deep gradients).
# float64 here avoids it; the small coarse OUTPUTS are cast to float32 only at save (no cancellation).
rho = gsw.rho(ts.salt, ts.temp, p_ref)
rho = ds.param.wet * rho + (1 - ds.param.wet) * RHO0           # BC fill, float64
ds.data['rho'] = rho
ds = DatasetCM26(ds.data, ds.param)                            # refresh state/grid with the new rho
print(f'DENSITY={DENSITY} factor {FACTOR} FGR{FGR} {SPLIT}: {ds.data.sizes["time"]} snapshots -> {OUT}', flush=True)

coarse = ds.compute_subfilter_forcing(factor=FACTOR, FGR_multiplier=FGR,
                                      coarsening=CoarsenKochkovMinMax(), filtering=Filtering(),
                                      percentile=PERCENTILE, add_rho_fluxes=True)
if not os.path.exists(f'{OUT}/param.nc'):
    depth_selector(coarse.param).to_netcdf(f'{OUT}/param.nc')
feat, feat_const = coarse.state.prepare_features()
feat = depth_selector(feat)
if not os.path.exists(f'{OUT}/permanent_features.nc'):
    depth_selector(feat_const).to_netcdf(f'{OUT}/permanent_features.nc')

# --- rolled-in N_buoyancy correction + coarse T/S (no separate fix step) ---
# The pipeline's Nsquared diffs in-situ rho across levels of VARYING pressure, so for the neutral
# density it is compression-contaminated. Recompute N the locally-referenced way (compression
# cancels): N^2 = g(-alpha(p) dtheta/dz + beta(p) dS/dz), alpha,beta = gsw at p_ref (0 for sigma0,
# local for neutral). Also store coarse T/S so any N^2 reference is recomputable + T/S is available
# as a potential input. Done per-snapshot (eager, one slice) -- matches the validated fix.
import types
fine_h = types.SimpleNamespace(param=ds.param)
coarse_h = types.SimpleNamespace(param=coarse.param)
filt, csn = Filtering(), CoarsenKochkovMinMax()
cg, cp = coarse.grid, coarse.param
dzB = cg.diff(cp.zl, 'Z')
pi = cg.interp(cp.zl.astype('float64'), 'Z')                     # interface pressure ~ depth (dbar)
p_ab = 0.0 if DENSITY == 'sigma0' else pi                        # alpha/beta reference pressure


def coarse_grain(field):                                         # fine -> coarse, same operators as rho
    _, _, fl = filt(None, None, field, fine_h, FACTOR * FGR)
    _, _, c = csn(None, None, fl, fine_h, coarse_h, factor=FACTOR)
    return c.compute()


def neutral_N(Tc, Sc):                                          # locally-referenced N on cell centers
    t, s = Tc.astype('float64').chunk({'zl': -1}), Sc.astype('float64').chunk({'zl': -1})
    ti, si = cg.interp(t, 'Z'), cg.interp(s, 'Z')
    alpha, beta = gsw.alpha(si, ti, p_ab), gsw.beta(si, ti, p_ab)
    N2 = np.maximum(9.8 * (-alpha * cg.diff(t, 'Z') / dzB + beta * cg.diff(s, 'Z') / dzB), 0.0) * cp.wet_w
    return cg.interp(np.sqrt(N2), 'Z') * cp.wet


for t in range(feat.sizes['time']):
    outfile = f'{OUT}/{SPLIT}-{t}.nc'
    if os.path.exists(outfile):
        try:
            with xr.open_dataset(outfile) as chk:
                # valid = finite coords AND a non-all-NaN data field (catches the all-NaN
                # quota-crash artifact, e.g. factor-15/train-67, that the yq-only check missed)
                if bool(np.isfinite(chk.yq.values).all()) and bool(np.isfinite(chk['Fx'].values).any()):
                    print(f'{SPLIT} [{t}] exists, skip', flush=True)
                    continue
        except Exception:
            pass
        os.remove(outfile)
    Tc, Sc = coarse_grain(ts.temp.isel(time=t)), coarse_grain(ts.salt.isel(time=t))
    Nc = neutral_N(Tc, Sc)
    snap = feat.isel(time=t)
    dims = snap['N_buoyancy'].dims
    snap['N_buoyancy'] = snap['N_buoyancy'].copy(data=Nc.transpose(*dims).values)   # correct, locally-ref N
    snap['temp'] = snap['N_buoyancy'].copy(data=Tc.transpose(*dims).values)
    snap['salt'] = snap['N_buoyancy'].copy(data=Sc.transpose(*dims).values)
    for v in snap.data_vars:                                     # float32 storage (compute was float64)
        snap[v] = snap[v].astype('float32')
    snap.to_netcdf(outfile)
    print(f'{SPLIT} [{t + 1}/{feat.sizes["time"]}] wrote {outfile}', flush=True)
print('DONE', flush=True)
print('DONE', flush=True)

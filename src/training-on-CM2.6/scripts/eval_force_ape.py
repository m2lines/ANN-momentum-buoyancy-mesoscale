"""Produce the forcing / APE-sink metrics (R2F_force, R2F_ape, + corr) for already-trained
rho-flux models, by re-running predict_ANN_rho().SGS_skill_rho() with the current cm26.py (the
metrics were added after these models' original skill-test). Each model is scored on its OWN
consistent test target (neutral -> subfilter-neutral; sigma0 -> subfilter), so the forcing
G = -u*.grad rho and APE-sink a = Upsilon.grad_h rho are built from that density's gradient + N^2
(Upsilon/clamps match MOM_meso_sfn_ANN.F90). Lets us compare sigma0 vs neutral on the quantities
the scheme actually applies, not just the flux. Writes skill-test-forceape/factor-*.nc.

ORDERING: do all the LIGHT factors (9,12,15) for every model FIRST so the comparison lands even if
the heavy factor-4 (largest grid, RAM-heavy forcing block) stalls; factor-4 runs LAST with time
subsampling (NTIME4). Skips factors already written.
  DEVICE cpu|cuda    NTIME4 (subsample for factor-4; default 8)
"""
import sys, os, gc
import numpy as np
import xarray as xr
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

FGR, STENCIL = 3, 3
DEVICE = os.environ.get('DEVICE', 'cpu')
NTIME4 = int(os.environ.get('NTIME4', 8))
MR = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR{FGR}')

MODELS = {  # path_save -> (model_file, subfilter, factors)
    'EXP_neutral_all4': (f'{MR}/EXP_neutral_all4/model/ann_instance.nc', 'subfilter-neutral', [9, 12, 15, 4]),
    'EXP_neutral': (f'{MR}/EXP_neutral/model/ann_instance.nc', 'subfilter-neutral', [9, 12, 4]),
    'EXP_sigma0':  (f'{MR}/EXP_sigma0/model/ann_instance.nc',  'subfilter',         [9, 12, 4]),
    'EXP0':        (f'{MR}/EXP0/model/ann_instance_20Dec.nc',  'subfilter',         [9, 12, 15, 4]),
}
KEYS = ['R2F_force_away', 'corr_F_force_away', 'R2F_ape_away', 'corr_F_ape_away', 'R2F_force', 'R2F_ape']

# build task list: every (model, factor) for light factors first (9,12,15), factor-4 last
tasks = [(m, f) for f in (9, 12, 15) for m, (_, _, facs) in MODELS.items() if f in facs]
tasks += [(m, 4) for m, (_, _, facs) in MODELS.items() if 4 in facs]

anns = {}
def get_ann(m):
    if m not in anns:
        anns[m] = import_ANN(MODELS[m][0]).to(DEVICE)
    return anns[m]

print(f'{"model":>12} {"fac":>4} | ' + ' '.join(f'{k:>17}' for k in KEYS), flush=True)
for m, fac in tasks:
    model_file, subfilter, _ = MODELS[m]
    out = f'{MR}/{m}/skill-test-forceape'; os.makedirs(out, exist_ok=True)
    fn = f'{out}/factor-{fac}.nc'
    if os.path.exists(fn):
        print(f'{m:>12} {fac:>4} |  (exists, skip)', flush=True); continue
    ds = read_datasets(['test'], [fac], subfilter=subfilter, FGR=FGR)[f'test-{fac}']
    if fac == 4 and NTIME4 > 0:                          # subsample time on the heavy finest grid
        ds = DatasetCM26(ds.data.isel(time=slice(0, NTIME4)), ds.param)
    skill = ds.predict_ANN_rho(get_ann(m), stencil_size=STENCIL, device=DEVICE).SGS_skill_rho()
    skill.to_netcdf(fn)
    vals = [float(skill[k].mean()) if k in skill else np.nan for k in KEYS]
    note = f'  (NT={NTIME4})' if fac == 4 else ''
    print(f'{m:>12} {fac:>4} | ' + ' '.join(f'{v:>17.4f}' for v in vals) + note, flush=True)
    del ds, skill; gc.collect()
print('\nDONE', flush=True)

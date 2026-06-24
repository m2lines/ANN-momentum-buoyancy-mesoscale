"""Matched evaluation of the density-gradient-kind pilot (see ../PLAN.md).

Each variant model must be scored with the SAME gradient kind it was trained on -- the auto
skill-test in train_script_rho_fluxes.py uses the production gradient for all, which is wrong
for sigma0c/neutral. Here we load each factor-9 model and evaluate it on the factor-9 test set
with its matching gradient (prod = existing coarse-grained sigma0; sigma0c / neutral from the
factor-9-rhokinds sidecar), then report depth-mean R2F / corr_F. The sigma0 flux target is the
same for all three, so the only thing that differs is the density-gradient INPUT.

  DEVICE cpu|cuda    FACTOR 9    SPLIT test
"""
import sys
import os
import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

FGR = 3
FACTOR = int(os.environ.get('FACTOR', 9))
SPLIT = os.environ.get('SPLIT', 'test')
STENCIL = int(os.environ.get('STENCIL', 3))
DEVICE = os.environ.get('DEVICE', 'cpu')
MODELROOT = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR{FGR}')
ROOT = os.path.expandvars(os.environ.get('CM26_DATA_ROOT', '/scratch/$USER/CM26_datasets/ocean3d'))
RKDIR = f'{ROOT}/subfilter/FGR{FGR}/factor-{FACTOR}-rhokinds'
OUT = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR{FGR}/rhokind_eval')
os.makedirs(OUT, exist_ok=True)
LOAD_VARS = ['Fx', 'Fy', 'rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']
VARIANTS = {'prod': None, 'sigma0c': ('rhox_sigma0c', 'rhoy_sigma0c'),
            'neutral': ('rhox_neutral', 'rhoy_neutral')}

base = read_datasets([SPLIT], [FACTOR])[f'{SPLIT}-{FACTOR}']
data0 = base.data[LOAD_VARS].load()
rk = xr.open_mfdataset(f'{RKDIR}/{SPLIT}-*.nc', combine='nested', concat_dim='time').sortby('time').load()
assert rk.sizes['time'] == data0.sizes['time'], 'rho-kinds time mismatch'
print(f'factor {FACTOR} {SPLIT}: {data0.sizes["time"]} snapshots, device {DEVICE}\n', flush=True)

print(f'{"variant":10s} {"R2F":>8s} {"corr_F":>8s}')
for v, g in VARIANTS.items():
    model = f'{MODELROOT}/rhokind_{v}{FACTOR}/model/ann_instance.nc'
    ann = import_ANN(model).to(DEVICE)
    data = data0.copy()
    if g is not None:                                   # swap in the variant gradient (prod = as-is)
        gx, gy = g
        data['rhox'] = (data['rhox'].dims, rk[gx].transpose(*data['rhox'].dims).values)
        data['rhoy'] = (data['rhoy'].dims, rk[gy].transpose(*data['rhoy'].dims).values)
    skill = DatasetCM26(data, base.param).predict_ANN_rho(ann, stencil_size=STENCIL, device=DEVICE).SGS_skill_rho()
    skill.to_netcdf(f'{OUT}/skill-{v}.nc')
    print(f'{v:10s} {float(skill.R2F.mean()):8.4f} {float(skill.corr_F.mean()):8.4f}', flush=True)
print('\nDONE', flush=True)

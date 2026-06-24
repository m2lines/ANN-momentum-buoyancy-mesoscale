"""Cross-eval for the density-gradient pilot: the ACTUAL online mismatch.

Online, MOM6 feeds the neutral (local-pressure) gradient to the sigma0-TRAINED model. So the
clean estimate of the online gradient-kind cost is: take the prod (sigma0-trained) model and
evaluate it on the neutral gradient vs on its native sigma0 gradient, depth-resolved. (The
neutral-TRAINED model is confounded by the sigma0-target/neutral-norm inconsistency, so it does
not answer this.) Reports R2F by depth band -- if the drop is confined to the deep ocean where
the flux is negligible, the online mismatch is benign where it matters (consistent with Tier-1).

  DEVICE cuda|cpu
"""
import sys
import os
import numpy as np
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

FGR, FACTOR, SPLIT, STENCIL = 3, 9, 'test', 3
DEVICE = os.environ.get('DEVICE', 'cuda')
MODELROOT = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR{FGR}')
ROOT = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d')
RKDIR = f'{ROOT}/subfilter/FGR{FGR}/factor-{FACTOR}-rhokinds'
LOAD_VARS = ['Fx', 'Fy', 'rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']

base = read_datasets([SPLIT], [FACTOR])[f'{SPLIT}-{FACTOR}']
data0 = base.data[LOAD_VARS].load()
rk = xr.open_mfdataset(f'{RKDIR}/{SPLIT}-*.nc', combine='nested', concat_dim='time').sortby('time').load()
ann = import_ANN(f'{MODELROOT}/rhokind_prod{FACTOR}/model/ann_instance.nc').to(DEVICE)  # sigma0-trained
print(f'prod (sigma0-trained) model on factor-{FACTOR} {SPLIT}, device {DEVICE}\n', flush=True)

bands = [(0, 500), (500, 1500), (1500, 5500)]
res = {}
for kind in ['prod', 'neutral']:
    data = data0.copy()
    if kind == 'neutral':
        data['rhox'] = (data['rhox'].dims, rk['rhox_neutral'].transpose(*data['rhox'].dims).values)
        data['rhoy'] = (data['rhoy'].dims, rk['rhoy_neutral'].transpose(*data['rhoy'].dims).values)
    skill = DatasetCM26(data, base.param).predict_ANN_rho(ann, stencil_size=STENCIL, device=DEVICE).SGS_skill_rho()
    res[kind] = skill
    zv = skill.zl.values
    print(f'[{kind:7s} input]  depth-mean R2F={float(skill.R2F.mean()):.4f}  corr_F={float(skill.corr_F.mean()):.4f}')

zv = res['prod'].zl.values
print('\n band-mean R2F:   prod-input  neutral-input   drop')
for lo, hi in bands:
    m = (zv >= lo) & (zv < hi)
    p = float(res['prod'].R2F[m].mean()); n = float(res['neutral'].R2F[m].mean())
    print(f'  {lo:4d}-{hi:4d} m:    {p:6.3f}      {n:6.3f}      {n - p:+.3f}')
print('\nDONE', flush=True)

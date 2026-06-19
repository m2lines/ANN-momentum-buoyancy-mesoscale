import sys, os
import numpy as np
sys.path.append('..')
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

# Canonical buoyancy ANN (committed in-repo): stencil 3, hidden [32,32], FGR3, EXP0.
ANN_PATH = os.environ.get('ANN_PATH',
    '/home/db194/ANN-momentum-buoyancy-mesoscale/CM26_ML_models/ocean3d/subfilter/FGR3/'
    'buoyancy/hidden-layer-32-32/seed-default/model/ann_instance_20Dec.nc')
FACTORS = eval(os.environ.get('FACTORS', '[9]'))
STENCIL = int(os.environ.get('STENCIL', '3'))
# SPLIT selects which dataset to score: 'test' (default), 'validate', or 'train'.
# Comparing R2F across the three is the overfitting check.
SPLIT = os.environ.get('SPLIT', 'test')
# NTIME>0 evenly subsamples the split to that many snapshots. predict_ANN_rho
# materializes the full (time,zl,y,x) fields, so peak memory scales with snapshot
# count; train (96/factor) OOMs at 128GB. Subsampling train to NTIME=24 matches
# test's count -> equal-N (apples-to-apples) overfitting comparison that fits in memory.
NTIME = int(os.environ.get('NTIME', '0'))
# DEVICE='cuda' runs the (now batched) inference on GPU. predict_ANN_rho does one
# batched forward per depth, so the GPU is well-fed -- ~10x vs CPU (unlike the old
# per-slice path, where GPU gave ~1.2x). 'cpu' (default) stays correct everywhere.
DEVICE = os.environ.get('DEVICE', 'cpu')
OUT = os.environ.get('SKILL_OUT', f'/scratch/db194/CM26_ML_models/FGR3/EXP0/skill-{SPLIT}-rho')
os.makedirs(OUT, exist_ok=True)

# Variables predict_ANN_rho + SGS_skill_rho read; preloading them to RAM removes the
# per-slice disk reads that otherwise dominate.
LOAD_VARS = ['Fx', 'Fy', 'rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']

ann = import_ANN(ANN_PATH).to(DEVICE)
ds = read_datasets([SPLIT], FACTORS)
for f in FACTORS:
    d = ds[f'{SPLIT}-{f}']
    data = d.data[LOAD_VARS]
    if NTIME and NTIME < data.sizes['time']:
        idx = np.unique(np.linspace(0, data.sizes['time'] - 1, NTIME).round().astype(int))
        data = data.isel(time=idx)
    d = DatasetCM26(data.load(), d.param)        # preload (subset) to RAM
    skill = d.predict_ANN_rho(ann, stencil_size=STENCIL, device=DEVICE).SGS_skill_rho()
    skill.to_netcdf(f'{OUT}/factor-{f}.nc')
    print('%-8s factor %2d:  R2F=%.4f  corr_F=%.4f' %
          (SPLIT, f, float(skill.R2F.mean()), float(skill.corr_F.mean())), flush=True)
print('DONE', flush=True)

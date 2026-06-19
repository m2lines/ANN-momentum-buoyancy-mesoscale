import sys, os
sys.path.append('..')
from helpers.cm26 import read_datasets
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
OUT = os.environ.get('SKILL_OUT', f'/scratch/db194/CM26_ML_models/FGR3/EXP0/skill-{SPLIT}-rho')
os.makedirs(OUT, exist_ok=True)

ann = import_ANN(ANN_PATH)
ds = read_datasets([SPLIT], FACTORS)
for f in FACTORS:
    skill = ds[f'{SPLIT}-{f}'].predict_ANN_rho(ann, stencil_size=STENCIL).SGS_skill_rho()
    skill.to_netcdf(f'{OUT}/factor-{f}.nc')
    print('%-8s factor %2d:  R2F=%.4f  corr_F=%.4f' %
          (SPLIT, f, float(skill.R2F.mean()), float(skill.corr_F.mean())), flush=True)
print('DONE', flush=True)

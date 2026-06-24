"""Score the new forcing- and APE-sink metrics (the two quantities the scheme actually applies,
from the flux-decomposition appendix) alongside the existing flux / horizontal-divergence metrics.
Loads factor-9 test once and evaluates the canonical EXP0 model on CPU. factor-9 only (canonical
resolution, representative). Prints depth-mean and surface coast-excluded (_away) R2 + correlation."""
import os, sys, xarray as xr
sys.path.insert(0, '/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

MODEL = os.environ.get('MODEL', '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP0/model/ann_instance_20Dec.nc')
LOAD = ['Fx', 'Fy', 'rhox', 'rhoy', 'N_buoyancy', 'sh_xy_h', 'sh_xx', 'rel_vort_h', 'delta_x']
ds = read_datasets(['test'], [9], subfilter='subfilter', FGR=3)['test-9']
d9 = DatasetCM26(ds.data[LOAD].load(), ds.param)
print('factor-9 test loaded', flush=True)

ann = import_ANN(MODEL).to('cpu')
skill = d9.predict_ANN_rho(ann, stencil_size=3, device='cpu').SGS_skill_rho()
skill.to_netcdf('/scratch/db194/mom6/CM26_ML_models/FGR3/EXP0/skill_forcing_ape_f9.nc')


def at(k, where):                                  # depth-mean or surface of a per-depth metric
    v = skill[k]
    if 'zl' not in v.dims:
        return float(v.values)
    return float((v.mean('zl') if where == 'mean' else v.isel(zl=0)).values)


print('\n%-22s %10s %10s' % ('metric (coast-excl)', 'depth-mean', 'surface'), flush=True)
for k in ['R2F_away', 'R2F_div_away', 'R2F_force_away', 'R2F_force_adv_away', 'R2F_ape_away',
          'corr_F_away', 'corr_F_div_away', 'corr_F_force_away', 'corr_F_force_adv_away', 'corr_F_ape_away']:
    print('%-22s %10.3f %10.3f' % (k, at(k, 'mean'), at(k, 'surf')), flush=True)
print('\nDONE', flush=True)

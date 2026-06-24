"""sigma0-vs-neutral denominator check for the forcing/APE metrics. The sigma0 EXP0 model's flux
predictions are kept FIXED; only the gradient (rho, rhox, rhoy) feeding the Ferrari construction is
swapped to locally-referenced (neutral) density -- what MOM6 actually uses online via
calc_isoneutral_slopes -- to see whether the denominator reference moves the forcing/APE skill.
factor-9 test; sigma0 and neutral share the grid and the 24 test times (verified)."""
import sys
sys.path.insert(0, '/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP0/model/ann_instance_20Dec.nc'
LOAD = ['Fx', 'Fy', 'rhox', 'rhoy', 'N_buoyancy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']
SWAP = ['N_buoyancy', 'rhox', 'rhoy']                              # gradient fields to swap sigma0 -> neutral

ds0 = read_datasets(['test'], [9], subfilter='subfilter', FGR=3)['test-9']
pred = DatasetCM26(ds0.data[LOAD].load(), ds0.param).predict_ANN_rho(
    import_ANN(MODEL).to('cpu'), stencil_size=3, device='cpu')      # sigma0 fluxes + sigma0 gradient
print('sigma0 prediction done', flush=True)

dsn = read_datasets(['test'], [9], subfilter='subfilter-neutral', FGR=3)['test-9'].data[SWAP].load()


def metrics(dset):
    s = dset.SGS_skill_rho()
    out = {}
    for k in ['R2F_force_away', 'R2F_ape_away', 'corr_F_force_away', 'corr_F_ape_away']:
        v = s[k]
        out[k] = (float(v.mean('zl')), float(v.isel(zl=0))) if 'zl' in v.dims else (float(v), float(v))
    return out


base = metrics(pred)                                                # sigma0 gradient (baseline)

dn = pred.data.copy()                                               # sigma0 fluxes, neutral gradient
for k in SWAP:
    dn[k] = (dn[k].dims, dsn[k].transpose(*dn[k].dims).values)
neut = metrics(DatasetCM26(dn, pred.param))                        # neutral gradient

print('\n%-22s %-19s %-19s' % ('metric (coast-excl)', 'sigma0-grad', 'neutral-grad'))
print('%-22s %9s %9s %9s %9s' % ('', 'dmean', 'surf', 'dmean', 'surf'))
for k in ['R2F_force_away', 'R2F_ape_away', 'corr_F_force_away', 'corr_F_ape_away']:
    b, n = base[k], neut[k]
    print('%-22s %9.3f %9.3f %9.3f %9.3f' % (k, b[0], b[1], n[0], n[1]), flush=True)
print('\nDONE', flush=True)

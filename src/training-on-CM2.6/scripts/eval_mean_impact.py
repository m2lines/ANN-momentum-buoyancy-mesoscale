"""Time-mean maps of the eddy impact on the mean flow and on the APE, diagnosed vs parameterized.

The offline-skill figures score INSTANTANEOUS agreement. What a climate model feels is the
time-mean effect: whether the scheme puts the right net forcing in the right places. This produces
those maps directly (no skill score -- the property plots are the point).

Two fields, both formed inside SGS_skill_rho with the MOM_meso_sfn_ANN.F90 limiters:
  forcing      G = -u* . grad rho   -> plotted as a DEPTH AVERAGE (plain mean over levels, which
                                       keeps the upper ocean prominent and so highlights the
                                       regions carrying the signal)
  APE sink     a = Upsilon . grad_h rho -> plotted as a DEPTH INTEGRAL, sum a*dz, the per-area
                                       quantity the online energy budget uses

Averaged over ALL splits (train+validate+test = 132 snapshots, ~11 years of monthly sampling across
model years 181-200) rather than test alone (24). This is a property plot, not a generalization
test, and train/validate/test skill agree to <=0.01, so pooling is legitimate -- state it in the
caption.

Streams one snapshot at a time and accumulates the 2-D reductions: predict_ANN_rho materializes
whole fields, so loading a split at once OOMs. Calling SGS_skill_rho on a single-time dataset
returns that snapshot's G/a through the audited path (its snapshot fields are isel(time=0)),
which is why nothing is reimplemented here.

  FACTORS=[9]  SPLITS=train,validate,test  MAXN=0 (0 = all)  DEVICE=cpu
  SAVE_PRED=1  ->  also persist Fx_pred/Fy_pred per snapshot under PRED/factor-N/
  USE_PRED=1   ->  re-read those instead of re-running inference (CPU-only; no GPU needed)
"""
import os, sys, glob, gc
import numpy as np
import gsw
import xarray as xr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

FGR, STENCIL = 3, 3
DEVICE = os.environ.get('DEVICE', 'cpu')
MAXN = int(os.environ.get('MAXN', 0))
SPLITS = os.environ.get('SPLITS', 'train,validate,test').split(',')
FACTORS = eval(os.environ.get('FACTORS', '[9]'))
MODEL = os.path.expandvars(
    '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc')
OUT = os.path.expandvars(os.environ.get('OUT',
    '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/mean-impact'))
SAVE_PRED = os.environ.get('SAVE_PRED', '0') == '1'
USE_PRED  = os.environ.get('USE_PRED', '0') == '1'   # re-read persisted predictions, skip inference
PRED = os.path.expandvars(os.environ.get('PRED',
    '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/predictions'))
os.makedirs(OUT, exist_ok=True)

ann = import_ANN(MODEL).to(DEVICE)

for fac in FACTORS:
    acc, n = {}, 0
    dz = None; area = None
    for split in SPLITS:
        ds_all = read_datasets([split], [fac], subfilter='subfilter-neutral', FGR=FGR)[f'{split}-{fac}']
        nt = ds_all.data.sizes['time']
        for it in range(nt if MAXN == 0 else min(nt, MAXN)):
            one = DatasetCM26(ds_all.data.isel(time=slice(it, it + 1)), ds_all.param)
            pfn = f'{PRED}/factor-{fac}/{split}-{it:03d}.nc'
            if USE_PRED and os.path.exists(pfn):
                # Re-read a previously persisted prediction instead of re-running the forward pass.
                # Assembles exactly what predict_ANN_rho returns (see cm26.py): the same small
                # Dataset of Fx/Fy/rhox/rhoy/N_buoyancy plus Fx_pred/Fy_pred, wrapped in a
                # DatasetCM26. The stored fields have no time axis, hence the [None, ...].
                data = xr.Dataset()
                keys = ['Fx', 'Fy', 'rhox', 'rhoy']
                if 'N_buoyancy' in one.data:
                    keys.append('N_buoyancy')
                for key in keys:
                    data[key] = one.nanvar(one.data[key]).copy(deep=True).compute()
                st = xr.open_dataset(pfn)
                data['Fx_pred'] = data['Fx'].copy(data=st['Fx_pred'].values[None, ...])
                data['Fy_pred'] = data['Fy'].copy(data=st['Fy_pred'].values[None, ...])
                st.close()
                sk = DatasetCM26(data, one.param).SGS_skill_rho()
            else:
                sk = one.predict_ANN_rho(ann, stencil_size=STENCIL, device=DEVICE).SGS_skill_rho()
            if dz is None:                                   # cell thickness from the level centres
                zl = sk['zl'].values
                dz = xr.DataArray(np.gradient(zl), dims=['zl'], coords={'zl': sk['zl']})
            if area is None:                                 # cell area for the horizontal averages
                area = (one.param.dxT * one.param.dyT)
            for k in ['G', 'G_pred']:                        # depth AVERAGE (plain mean over levels)
                acc[k] = acc.get(k, 0.) + sk[k].mean('zl', skipna=True)
            for k in ['a', 'a_pred']:                        # depth INTEGRAL, sum a*dz
                acc[k] = acc.get(k, 0.) + (sk[k] * dz).sum('zl', skipna=True)
            # ...and keep the VERTICAL structure too: area-weighted horizontal mean at each level,
            # which says where in the column the exchange actually happens. Cheap to carry (1-D per
            # snapshot) and not recoverable from the depth-collapsed maps above.
            for k in ['G', 'a']:
                for kk in (k, k + '_pred'):
                    w = area.where(np.isfinite(sk[kk]))
                    acc[kk + '_prof'] = acc.get(kk + '_prof', 0.) + \
                        (sk[kk] * area).sum(['yh', 'xh']) / w.sum(['yh', 'xh'])
            # APE sink integrated only BELOW the mixed layer. Online, MOM6 tapers the streamfunction
            # toward the surface (Ferrari/FGNV) and mixed-layer restratification is the job of a
            # separate submesoscale closure, so the near-surface part of the column is not what this
            # scheme is asked to deliver -- integrating it in flatters or flatters-not the comparison
            # for the wrong reasons. MLD by de Boyer Montegut: shallowest level whose sigma0 exceeds
            # the shallowest-level value by 0.03 kg/m3. sigma0 must be computed here from salt/temp:
            # the stored `rho` is LOCALLY referenced, so its vertical increase carries compression
            # and a threshold applied to it would give a spuriously shallow mixed layer.
            sig = 1000. + gsw.sigma0(one.data.salt.isel(time=0), one.data.temp.isel(time=0))
            exc = (sig - sig.isel(zl=0)) > 0.03
            mld = xr.where(exc.any('zl'), exc.idxmax('zl'), float(sk['zl'].max()))
            below = sk['zl'] > mld
            # Same treatment for the forcing: the tapering argument applies to G no less than to a,
            # and carrying both on one convention keeps the two rows of the figure comparable.
            for kk in ['a', 'a_pred', 'G', 'G_pred']:
                acc[kk + '_belowml'] = acc.get(kk + '_belowml', 0.) + \
                    (sk[kk] * dz).where(below).sum('zl')
            for kk in ['G', 'G_pred']:                       # ...and a below-ML depth AVERAGE for G,
                w = dz.where(below & np.isfinite(sk[kk]))     # to match the full-column row's units
                acc[kk + '_belowml_avg'] = acc.get(kk + '_belowml_avg', 0.) + \
                    (sk[kk] * dz).where(below).sum('zl') / w.sum('zl')
            acc['mld'] = acc.get('mld', 0.) + mld
            # Persist the predicted flux. This is the expensive object -- the ANN forward pass is
            # ~80% of the per-snapshot cost -- and every analysis that has wanted it so far has
            # recomputed it from scratch and thrown it away. Saved once, any later diagnostic
            # (Upsilon, G, a, any depth slice, any other reduction) is a cheap re-read: the source
            # data already carries rhox/rhoy/N_buoyancy, so Fx_pred/Fy_pred completes the set.
            # ~24 MB/snapshot at factor-9, ~120 MB at factor-4; ~25 GB for all four factors.
            if SAVE_PRED:
                pdir = f'{PRED}/factor-{fac}'
                os.makedirs(pdir, exist_ok=True)
                pfn = f'{pdir}/{split}-{it:03d}.nc'
                if not os.path.exists(pfn):
                    enc = {k: {'zlib': True, 'complevel': 4} for k in ['Fx_pred', 'Fy_pred']}
                    xr.Dataset({'Fx_pred': sk['Fx_pred'], 'Fy_pred': sk['Fy_pred']}).to_netcdf(
                        pfn, encoding=enc)
            n += 1
            if n % 10 == 0:
                print(f'  factor-{fac}: {n} snapshots', flush=True)
            del one, sk; gc.collect()
        del ds_all; gc.collect()

    out = xr.Dataset({k: v / n for k, v in acc.items()})
    out['n_snapshots'] = n
    out.attrs['splits'] = ','.join(SPLITS)
    out.attrs['note'] = ('G is a plain depth average over levels; a is the depth integral sum(a*dz). '
                         'Time mean over all listed splits -- a property plot, not a skill test.')
    fn = f'{OUT}/factor-{fac}.nc'
    out.to_netcdf(fn)
    print(f'factor-{fac}: wrote {fn} from {n} snapshots', flush=True)

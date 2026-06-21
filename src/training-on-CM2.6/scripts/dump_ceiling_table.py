"""Phase 0 of the predictability-ceiling study (see ../CEILING_STUDY_PLAN.md).

Dump a held-out table of (normalized input X, dimensionless target T, prediction P,
per-point scale s, and a few normalized-OUT magnitudes) for the trained baseline ANN
on the test split. Everything downstream (the nearest-neighbour ceiling B1 and the
residual-structure analysis B2) reads this one npz -- no model or data re-reads.

The ANN regresses  X (45-dim, what it actually sees) -> T (2-dim dimensionless flux),
and the dimensional flux is  F = T * s  with  s = rho_norm * gradv_norm * delta_x**2
(on wet points). The reported R2F lives in the dimensional F space, so we keep s to
re-dimensionalize the conditional-variance estimate and make B1 comparable to ~0.73.

Points are subsampled per (factor, depth) so the table and the KDTree stay tractable;
all test times are pooled. Run in the Pavel container (torch).

  NPERLEVEL  points kept per (factor, depth)   [default 10000]
  FACTORS    comma list                         [default 4,9,12,15]
  MODEL      ann_instance.nc                     [default baseline st3_h32-32_s0]
  OUT        output npz                          [default scratch ceiling/table.npz]
"""
import os
import sys
import numpy as np
import xarray as xr
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import import_ANN

NPERLEVEL = int(os.environ.get('NPERLEVEL', 10000))
NTIME = int(os.environ.get('NTIME', 0))   # 0 = all test times; else evenly-spaced subsample
FACTORS = [int(f) for f in os.environ.get('FACTORS', '4,9,12,15').split(',')]
SCRATCH = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity')
MODEL = os.environ.get('MODEL', f'{SCRATCH}/st3_h32-32_s0/model/ann_instance.nc')
OUT = os.path.expandvars(os.environ.get('OUT', '/scratch/$USER/mom6/CM26_ML_models/FGR3/ceiling/table.npz'))
STENCIL = 3
DEPTH_IDX = np.arange(0, 50, 2)   # the model's training depths (configuration.txt)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
rng = np.random.default_rng(0)
ann = import_ANN(MODEL)
print(f'loaded {MODEL}: layers {list(ann.layer_sizes)}', flush=True)

# only the fields the inference + the B2 diagnostics touch
LOAD_VARS = ['rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'Fx', 'Fy', 'delta_x', 'u', 'v']
ds = read_datasets(['test'], FACTORS, FGR=3)

cols = {k: [] for k in ['X', 'Tx', 'Ty', 'Px', 'Py', 'Fx', 'Fy', 's',
                        'rho_norm', 'gradv_norm', 'KE', 'factor', 'zl', 'depth', 'tt', 'pos']}

for factor in FACTORS:
    d = ds[f'test-{factor}']
    have = [v for v in LOAD_VARS if v in d.data]
    data = d.data[have]
    if NTIME and NTIME < len(data.time):
        data = data.isel(time=np.linspace(0, len(data.time) - 1, NTIME).round().astype(int))
    # keep lazy (chunked zl=1,time=1): each 2D slice reads on demand in inference. Preloading
    # the whole factor OOMs at factor-4 (~10 GB); we only touch NTIME x len(DEPTH_IDX) slices.
    d = DatasetCM26(data, d.param)
    ntime = len(d.data.time)
    print(f'factor {factor}: {ntime} test times, {len(DEPTH_IDX)} depths', flush=True)

    for zl in DEPTH_IDX:
        depth = float(d.data.zl.values[zl])
        wet = d.param.wet.isel(zl=zl).values > 0.5         # (y, x)
        Xs, Ts, Ps, Fs, ss, kes, rns, gns, tts, poss = [], [], [], [], [], [], [], [], [], []

        for t in range(ntime):
            sl = DatasetCM26(d.data.isel(time=t, zl=zl), d.param.isel(zl=zl))
            with torch.no_grad():
                pred = sl.state.ANN_rho_inference(ann, stencil_size=STENCIL, return_features=True)
            X = pred['input_features'].cpu().numpy()        # (Npts, 45)
            rn = pred['rho_norm'].cpu().numpy().reshape(-1)
            gn = pred['gradv_norm'].cpu().numpy().reshape(-1)
            s = (rn * gn * pred['cell'].cpu().numpy().reshape(-1))   # = rho_norm*gradv_norm*delta_x^2
            Fxp = pred['Fx'].cpu().numpy().reshape(-1)
            Fyp = pred['Fy'].cpu().numpy().reshape(-1)
            Fxt = np.nan_to_num(sl.data.Fx.values).reshape(-1)
            Fyt = np.nan_to_num(sl.data.Fy.values).reshape(-1)
            m = wet.reshape(-1) & np.isfinite(s) & (np.abs(s) > 0)
            # KE = 0.5(u^2+v^2) -- a magnitude the non-dim discards; 0 if u/v absent
            if 'u' in sl.data and 'v' in sl.data:
                ke = 0.5 * (np.nan_to_num(sl.data.u.values)**2 + np.nan_to_num(sl.data.v.values)**2).reshape(-1)
            else:
                ke = np.zeros_like(s)
            Xs.append(X[m]); ss.append(s[m]); kes.append(ke[m]); rns.append(rn[m]); gns.append(gn[m])
            Ts.append(np.stack([Fxt[m] / s[m], Fyt[m] / s[m]], 1))    # dimensionless target
            Ps.append(np.stack([Fxp[m] / s[m], Fyp[m] / s[m]], 1))    # network output (P_T)
            Fs.append(np.stack([Fxt[m], Fyt[m]], 1))                  # dimensional truth
            poss.append(np.flatnonzero(m))                            # flattened spatial id (const across t)
            tts.append(np.full(int(m.sum()), t))                      # time snapshot index

        if not Xs:
            continue
        X = np.concatenate(Xs); T = np.concatenate(Ts); P = np.concatenate(Ps)
        F = np.concatenate(Fs); s = np.concatenate(ss); ke = np.concatenate(kes)
        rn = np.concatenate(rns); gn = np.concatenate(gns)
        tt = np.concatenate(tts); pos = np.concatenate(poss)
        n = X.shape[0]
        idx = rng.choice(n, min(NPERLEVEL, n), replace=False)
        X, T, P, F, s, ke, rn, gn = X[idx], T[idx], P[idx], F[idx], s[idx], ke[idx], rn[idx], gn[idx]
        tt, pos = tt[idx], pos[idx]

        # B2 probes = magnitudes the non-dimensionalization normalizes OUT of the inputs:
        # s (= rho_norm * gradv_norm * delta_x^2, the gradient/strain magnitudes) and KE.
        # If the residual T-P correlates with these, the non-dim is discarding usable signal.
        cols['X'].append(X.astype('float32'))
        cols['Tx'].append(T[:, 0].astype('float32')); cols['Ty'].append(T[:, 1].astype('float32'))
        cols['Px'].append(P[:, 0].astype('float32')); cols['Py'].append(P[:, 1].astype('float32'))
        cols['Fx'].append(F[:, 0].astype('float32')); cols['Fy'].append(F[:, 1].astype('float32'))
        cols['s'].append(s.astype('float32')); cols['KE'].append(ke.astype('float32'))
        cols['rho_norm'].append(rn.astype('float32')); cols['gradv_norm'].append(gn.astype('float32'))
        cols['factor'].append(np.full(len(idx), factor, 'int16'))
        cols['zl'].append(np.full(len(idx), zl, 'int16'))
        cols['depth'].append(np.full(len(idx), depth, 'float32'))
        cols['tt'].append(tt.astype('int16')); cols['pos'].append(pos.astype('int32'))
        print(f'  zl {zl:2d} ({depth:7.1f} m): kept {len(idx)} of {n}', flush=True)

    del d
    ds[f'test-{factor}'] = None

out = {k: np.concatenate(v) for k, v in cols.items() if k != 'X'}
out['X'] = np.concatenate(cols['X'])
np.savez(OUT, **out)
print(f'wrote {OUT}: {out["X"].shape[0]} rows, X {out["X"].shape}', flush=True)

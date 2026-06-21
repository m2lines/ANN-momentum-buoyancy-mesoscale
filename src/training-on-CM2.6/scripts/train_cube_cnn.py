"""Phase A2 of the predictability-ceiling study (see ../CEILING_STUDY_PLAN.md).

A 3-D "cube" closure as the from-below ceiling probe: predict the horizontal density flux
(Fx, Fy) at every point from the FULL local water column of coarse fields over a large
horizontal footprint -- a strict superset of the per-point 3x3 MLP. If this maximal-context
model does not beat the MLP on held-out data, we are at the local predictability ceiling; if
it jumps, we were footprint / vertical / stratification limited.

Architecture (CubeCNN): 2-D CNN over (y, x); the full depth column of each input field is
stacked into channels, so every output level sees the whole column (vertical coupling via 1x1
channel mixing) and a dilated 3x3 stack gives a large horizontal receptive field (the cube
footprint). Output is a dimensionless (Fx, Fy) per depth, redimensionalized by the physical
scale s = |grad rho|*|grad u|*dx^2 (so the net emits O(1), not the surface->abyss range).
Scored through the SAME SGS_skill_rho as the MLP -> apples-to-apples R2F.

MULTI-RESOLUTION (option A): pool all factors so the cube gets the data/diversity the MLP had.
The grid is supplied to the network three ways:
  1. output scale s carries dx^2 per factor (magnitude) -- automatic;
  2. inputs standardized PER FACTOR (each resolution's fields made comparable);
  3. an explicit resolution channel  log(dx / deformation_radius)  -- "how resolved are the
     eddies" -- standardized GLOBALLY so the cross-factor signal survives.
Crops are a fixed CROPxCROP, so snapshots from different-sized grids batch together. Data lives
on CPU (factor-4 is too big for GPU); crop batches move to GPU per step.

Data-limitation is the trap: report TRAIN and TEST R2F; if train >> test the ceiling estimate
is invalid (overfit, not info-limited).

  FACTORS=4:9:12:15  EPOCHS=300  CROP=128  BATCH=8  WIDTH=128  DILATIONS=1,2,4,8
  MAXTRAIN=24 (per factor; 0=all)  FIELDS=... (':'-sep)  OUT=.../ceiling/cube/pooled
"""
import os
import re
import sys
import json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from time import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.cm26 import read_datasets, DatasetCM26

FACTORS = [int(x) for x in re.split('[,:]', os.environ.get('FACTORS', '4:9:12:15'))]
EPOCHS = int(os.environ.get('EPOCHS', 300))
CROP = int(os.environ.get('CROP', 128))
BATCH = int(os.environ.get('BATCH', 8))
WIDTH = int(os.environ.get('WIDTH', 128))
DIL = [int(x) for x in re.split('[,:]', os.environ.get('DILATIONS', '1,2,4,8'))]
LR = float(os.environ.get('LR', 1e-3))
WD = float(os.environ.get('WD', 1e-5))
MAXTRAIN = int(os.environ.get('MAXTRAIN', 24))     # per-factor train snapshot cap (bounds RAM); 0=all
MAXTEST = int(os.environ.get('MAXTEST', 12))       # per-factor test snapshot cap for eval; 0=all
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
SEED = int(os.environ.get('SEED', 0))
OUT = os.path.expandvars(os.environ.get('OUT', '/scratch/$USER/mom6/CM26_ML_models/FGR3/ceiling/cube/pooled'))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

FIELDS = re.split('[,:]', os.environ.get('FIELDS', 'rhox,rhoy,sh_xx,sh_xy_h,rel_vort_h,u_h,v_h,'
                  'rho,N_buoyancy,rho_grad_mag,shear_vort_mag'))   # ':' for comma-safe --export
SCALE_FIELDS = ['rho_grad_mag', 'shear_vort_mag']    # output scale ~ |grad rho|*|grad u| (* dx^2)
NONDIM = int(os.environ.get('NONDIM', 1))            # 1 = option B: MLP-style per-point non-dim
rng = np.random.default_rng(SEED)


def nan0(x):
    v = x.values if hasattr(x, 'values') else x
    return np.where(np.isfinite(v), v, 0.0).astype('float32')


def stack_fields(dt, fields):
    return np.concatenate([nan0(dt[f]) for f in fields], 0)       # (n_fields*zl, Y, X)


def stack_nondim(dt):
    """Option B: the MLP's per-point non-dimensionalization. Normalize each gradient group by
    its OWN local magnitude so the inputs are scale-free DIRECTIONS (exactly the MLP's inputs),
    plus velocity direction and stratification. The output is still redimensionalized by
    s=|grad rho|*|grad u|*dx^2, so directions-in / magnitude-out matches the MLP. 8 blocks x zl."""
    g = np.maximum(nan0(dt['rho_grad_mag']), 1e-30)
    h = np.maximum(nan0(dt['shear_vort_mag']), 1e-30)
    spd = np.sqrt(nan0(dt['u_h']) ** 2 + nan0(dt['v_h']) ** 2) + 1e-30
    blocks = [nan0(dt['rhox']) / g, nan0(dt['rhoy']) / g,                       # grad-rho direction
              nan0(dt['sh_xx']) / h, nan0(dt['sh_xy_h']) / h, nan0(dt['rel_vort_h']) / h,  # strain/vort dir
              nan0(dt['u_h']) / spd, nan0(dt['v_h']) / spd,                     # velocity direction
              nan0(dt['N_buoyancy'])]                                           # stratification (standardized later)
    return np.concatenate(blocks, 0)                 # (8*zl, Y, X)


def res_channel(d):
    """log(dx / deformation_radius): the resolution / 'how resolved' signal (2D, Y,X)."""
    dx = np.sqrt(d.param.dxT.values * d.param.dyT.values).astype('float32')
    dr = d.data['deformation_radius']
    for dim in ('time', 'zl'):
        if dim in dr.dims:
            dr = dr.mean(dim)
    dr = np.nan_to_num(dr.values, nan=0.0).astype('float32')
    return np.log(dx / np.where(dr > 1.0, dr, np.nan))    # nan where dr invalid -> handled later


def load_split(data, wet, dx2, maxn):
    nt = len(data.time)
    if maxn and maxn < nt:
        data = data.isel(time=np.linspace(0, nt - 1, maxn).round().astype(int))
        nt = maxn
    Xs, FXs, FYs, Ss = [], [], [], []
    for t in range(nt):
        dt = data.isel(time=t)
        Xs.append(stack_nondim(dt) if NONDIM else stack_fields(dt, FIELDS))
        FXs.append(np.nan_to_num(dt.Fx.values).astype('float32'))
        FYs.append(np.nan_to_num(dt.Fy.values).astype('float32'))
        s = np.ones_like(FXs[-1])
        for sf in SCALE_FIELDS:
            s = s * np.nan_to_num(dt[sf].values).astype('float32')
        Ss.append(np.maximum(s * dx2[None], 0.0))
    return np.stack(Xs), np.stack(FXs), np.stack(FYs), np.stack(Ss), data


class CubeCNN(nn.Module):
    def __init__(self, c_in, z, width, dilations):
        super().__init__()
        self.z = z
        layers = [nn.Conv2d(c_in, width, 1), nn.ReLU()]
        for dl in dilations:
            layers += [nn.Conv2d(width, width, 3, padding=dl, dilation=dl), nn.ReLU()]
        layers += [nn.Conv2d(width, 2 * z, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y[:, :self.z], y[:, self.z:]


def flux_norm(Fx, Fy, wetb):
    num = ((Fx ** 2 + Fy ** 2) * wetb).sum((-1, -2))
    cnt = wetb.sum((-1, -2)).clamp(min=1)
    return (1.0 / torch.sqrt((num / cnt).clamp(min=1e-30)))[..., None, None]   # (B,Z,1,1)


def main():
    print(f'factors {FACTORS} | device {DEVICE} | width {WIDTH} dil {DIL} | crop {CROP} batch {BATCH} '
          f'| maxtrain/factor {MAXTRAIN}', flush=True)
    dev = torch.device(DEVICE)
    ds = read_datasets(['train', 'test'], FACTORS, FGR=3)

    # ---- load every factor to CPU, standardize physical fields PER FACTOR ----
    FA = {}
    res_pool = []
    for fa in FACTORS:
        d = ds[f'test-{fa}']            # same param for train/test
        wet = (d.param.wet.values > 0.5).astype('float32')
        dx2 = (d.param.dxT.values * d.param.dyT.values).astype('float32')
        res = res_channel(d)            # (Y,X), may contain nan on land
        Xtr, FXtr, FYtr, Str, _ = load_split(ds[f'train-{fa}'].data, wet, dx2, MAXTRAIN)
        Xte, FXte, FYte, Ste, te_xr = load_split(d.data, wet, dx2, MAXTEST)
        C = Xtr.shape[1]
        wet_c = np.repeat(wet[None], C // wet.shape[0], 0).reshape(C, *wet.shape[1:])[None]
        denom = np.maximum(wet_c.sum((0, 2, 3)) * len(Xtr), 1)
        mu = ((Xtr * wet_c).sum((0, 2, 3)) / denom).astype('float32')[None, :, None, None]
        sd = (np.sqrt(((Xtr - mu) ** 2 * wet_c).sum((0, 2, 3)) / denom).astype('float32') + 1e-8)[None, :, None, None]

        def nrm_(X):       # standardize in place (no doubling of the big arrays), zero land
            X -= mu; X /= sd; X *= wet_c
            return X
        FA[fa] = dict(Xtr=nrm_(Xtr), FXtr=FXtr, FYtr=FYtr, Str=Str,
                      Xte=nrm_(Xte), FXte=FXte, FYte=FYte, Ste=Ste,
                      wet=wet, res=res, te_xr=te_xr, param=d.param, tr_xr=ds[f'train-{fa}'].data)
        res_pool.append(res[np.isfinite(res)])
        print(f'  factor {fa}: train {FA[fa]["Xtr"].shape} test {FA[fa]["Xte"].shape}', flush=True)

    # ---- global resolution-channel standardization (keeps the cross-factor signal) ----
    rp = np.concatenate(res_pool); gmu, gsd = float(rp.mean()), float(rp.std()) + 1e-8
    Z = FA[FACTORS[0]]['wet'].shape[0]
    for fa in FACTORS:
        r = (FA[fa]['res'] - gmu) / gsd
        r = np.where(np.isfinite(r), r, 0.0).astype('float32')[None]          # (1,Y,X), land/nan->0
        for k in ('Xtr', 'Xte'):
            n = FA[fa][k].shape[0]
            FA[fa][k] = np.concatenate([FA[fa][k], np.repeat(r[None], n, 0)], 1)   # append res channel
    C = FA[FACTORS[0]]['Xtr'].shape[1]
    print(f'Z={Z} C={C} (incl. resolution channel)', flush=True)

    model = CubeCNN(C, Z, WIDTH, DIL).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f'CubeCNN params: {nparam:,}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [EPOCHS // 2, EPOCHS * 3 // 4], gamma=0.1)

    # crop index over all factors (pooled). steps/epoch ~ total snapshots / batch.
    index = [(fa, t) for fa in FACTORS for t in range(len(FA[fa]['Xtr']))]
    nstep = max(1, len(index) // BATCH)

    def grab_crop(fa, t):
        a = FA[fa]; Y, Xw = a['wet'].shape[1:]
        if CROP and CROP < min(Y, Xw):
            y0, x0 = rng.integers(0, Y - CROP + 1), rng.integers(0, Xw - CROP + 1)
            ys, xs = slice(y0, y0 + CROP), slice(x0, x0 + CROP)
        else:
            ys, xs = slice(None), slice(None)
        return (a['Xtr'][t, :, ys, xs], a['FXtr'][t, :, ys, xs], a['FYtr'][t, :, ys, xs],
                a['Str'][t, :, ys, xs], a['wet'][:, ys, xs])

    t0 = time()
    for ep in range(EPOCHS):
        model.train()
        order = [index[i] for i in rng.permutation(len(index))]
        ep_loss = 0.0; nb = 0
        for i in range(0, len(order), BATCH):
            batch = order[i:i + BATCH]
            Xc, FXc, FYc, Sc, Wc = zip(*[grab_crop(fa, t) for fa, t in batch])
            Xb = torch.tensor(np.stack(Xc), device=dev)
            FXb = torch.tensor(np.stack(FXc), device=dev); FYb = torch.tensor(np.stack(FYc), device=dev)
            Sb = torch.tensor(np.stack(Sc), device=dev); Wb = torch.tensor(np.stack(Wc), device=dev)
            pfx, pfy = model(Xb); pfx, pfy = pfx * Sb, pfy * Sb
            fn = flux_norm(FXb, FYb, Wb)
            d2 = ((pfx - FXb) ** 2 + (pfy - FYb) ** 2) * (fn ** 2) * Wb
            loss = d2.sum() / Wb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        sched.step()
        if (ep + 1) % max(1, EPOCHS // 20) == 0:
            print(f'  ep {ep+1:4d}/{EPOCHS}  train_loss {ep_loss/nb:.5f}  ({time()-t0:.0f}s)', flush=True)

    torch.save(model.state_dict(), f'{OUT}/cube_cnn.pt')
    json.dump(dict(factors=FACTORS, fields=FIELDS, width=WIDTH, dilations=DIL, crop=CROP, epochs=EPOCHS,
                   lr=LR, wd=WD, maxtrain=MAXTRAIN, nparam=nparam, seed=SEED), open(f'{OUT}/config.json', 'w'))

    # ---- eval per factor through SGS_skill_rho (same metric as the MLP) ----
    def predict(Xnp, Snp):
        model.eval(); ox, oy = [], []
        with torch.no_grad():
            for t in range(len(Xnp)):
                pfx, pfy = model(torch.tensor(Xnp[t:t+1], device=dev))
                S = torch.tensor(Snp[t:t+1], device=dev)
                ox.append((pfx * S)[0].cpu().numpy()); oy.append((pfy * S)[0].cpu().numpy())
        return np.stack(ox), np.stack(oy)

    def skill(split_xr, Xnp, Snp, param):
        pfx, pfy = predict(Xnp, Snp)
        msk = np.broadcast_to(param.wet.values > 0.5, pfx.shape)
        data = xr.Dataset()
        for k in ['Fx', 'Fy', 'rhox', 'rhoy']:
            data[k] = split_xr[k]
        data['Fx_pred'] = data['Fx'].copy(data=np.where(msk, pfx, np.nan))
        data['Fy_pred'] = data['Fy'].copy(data=np.where(msk, pfy, np.nan))
        return DatasetCM26(data, param).SGS_skill_rho()

    print('\n=== R2F by factor (cube, multi-resolution) ===', flush=True)
    summary = {}
    for fa in FACTORS:
        a = FA[fa]
        r2_te = skill(a['te_xr'], a['Xte'], a['Ste'], a['param']).R2F.values
        tr_xr = a['tr_xr']
        if MAXTRAIN and MAXTRAIN < len(tr_xr.time):
            tr_xr = tr_xr.isel(time=np.linspace(0, len(tr_xr.time) - 1, MAXTRAIN).round().astype(int))
        r2_tr = skill(tr_xr, a['Xtr'], a['Str'], a['param']).R2F.values
        depth = a['te_xr'].zl.values; up = depth < 1500
        summary[fa] = dict(all_tr=float(np.nanmean(r2_tr)), all_te=float(np.nanmean(r2_te)),
                           up_tr=float(np.nanmean(r2_tr[up])), up_te=float(np.nanmean(r2_te[up])))
        np.save(f'{OUT}/r2f_test_f{fa}.npy', r2_te); np.save(f'{OUT}/r2f_train_f{fa}.npy', r2_tr)
        s = summary[fa]
        print(f'  factor {fa:2d}:  <1500m  train {s["up_tr"]:.3f}  test {s["up_te"]:.3f}    '
              f'all-depth  train {s["all_tr"]:.3f}  test {s["all_te"]:.3f}', flush=True)
    json.dump(summary, open(f'{OUT}/summary.json', 'w'))
    print(f'\nsaved -> {OUT}', flush=True)


if __name__ == '__main__':
    main()

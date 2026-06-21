"""Phase B1 of the predictability-ceiling study (see ../CEILING_STUDY_PLAN.md).

Model-free estimate of the irreducible noise floor of the local 3x3 closure, via the
Gamma test (Stefansson et al. 1997): for k nearest neighbours in the (raw, dimensional)
local-state space, the mean-squared flux difference extrapolated to zero input-distance
estimates the variance of the part of F NOT explained by ANY smooth function of the local
state -- the irreducible floor for a deterministic local closure.

Two confounds, both handled:
1. Heavy tails: we work in the DIMENSIONAL flux space the ANN is trained/scored in (uncentered
   R2F), not the dimensionless T=F/s space (dividing by near-zero s blows up the target).
2. Field autocorrelation: the nearest neighbour in input space is often a SPATIAL/temporal
   neighbour whose flux is correlated for reasons unrelated to the inputs (adjacent cells even
   share stencil values) -> artificially small flux-gap -> inflated ceiling. We forbid
   neighbours from the same snapshot (tt) and the same location (pos): only cross-time,
   cross-location pairs count.

Conditioning input = raw 45-dim local stencil, reduced to its ~VAR_KEEP PCA subspace (sparse
45-dim -> far NNs -> noisy; PCA -> denser, stable; smaller class => conservative/lower ceiling).

Per (factor, depth) we report:
  R2F_model      = 1 - <|F-Fpred|^2>/<|F|^2>          (the trained ANN, == reported R2F)
  R2F_ceil_lb    = 1 - Gamma_k1/<|F|^2>               (conservative: nearest cross-time NN, no
                                                        extrapolation -> overestimates noise)
  R2F_ceiling    = 1 - Gamma_extrap/<|F|^2>           (Gamma test extrapolated to zero distance)
The truth sits between lb and the extrapolation. If even R2F_ceil_lb > R2F_model, the headroom
is decisive.  IN=table.npz  K=8 (extrap neighbours)  KQ=40 (candidates)  VAR_KEEP=0.95.
"""
import os
import numpy as np
from scipy.spatial import cKDTree

IN = os.path.expandvars(os.environ.get('IN', '/scratch/$USER/mom6/CM26_ML_models/FGR3/ceiling/table.npz'))
K = int(os.environ.get('K', 8))
KQ = int(os.environ.get('KQ', 40))
VAR_KEEP = float(os.environ.get('VAR_KEEP', 0.95))
d = np.load(IN)
X, rn, gn = d['X'], d['rho_norm'], d['gradv_norm']
F = np.stack([d['Fx'], d['Fy']], 1)
Fpred = np.stack([d['Px'] * d['s'], d['Py'] * d['s']], 1)
factor, zl, depth, tt, pos = d['factor'], d['zl'], d['depth'], d['tt'], d['pos']
U = np.concatenate([X[:, :18] * rn[:, None], X[:, 18:] * gn[:, None]], 1)   # raw dimensional stencil
print(f'{IN}: {X.shape[0]} rows, U {U.shape}, K={K} KQ={KQ} VAR_KEEP={VAR_KEEP}', flush=True)


def gamma_test(Ug, Fg, ttg, posg):
    n = Ug.shape[0]
    if n < KQ + 5:
        return np.nan, np.nan
    Us = (Ug - Ug.mean(0)) / (Ug.std(0) + 1e-30)
    u, sv, _ = np.linalg.svd(Us - Us.mean(0), full_matrices=False)
    keep = max(2, int(np.searchsorted(np.cumsum(sv ** 2) / (sv ** 2).sum(), VAR_KEEP) + 1))
    Z = u[:, :keep] * sv[:keep]
    dist, idx = cKDTree(Z).query(Z, k=KQ + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]                       # drop self
    # keep only cross-time AND cross-location neighbours (kill field autocorrelation)
    valid = (ttg[idx] != ttg[:, None]) & (posg[idx] != posg[:, None])
    gap = ((Fg[idx] - Fg[:, None, :]) ** 2).sum(-1)          # |F_i - F_j|^2, (n, KQ)
    dist_f = np.where(valid, dist, np.inf)
    order = np.argsort(dist_f, axis=1)[:, :K]                 # first K qualifying, by distance
    dist_k = np.take_along_axis(dist_f, order, 1)
    gap_k = np.take_along_axis(gap, order, 1)
    fin = np.isfinite(dist_k)
    delta = np.array([(dist_k[:, k][fin[:, k]] ** 2).mean() for k in range(K)])
    gamma = np.array([0.5 * gap_k[:, k][fin[:, k]].mean() for k in range(K)])
    G_lb = float(gamma[0])                                    # k=1 cross-time, no extrapolation
    G_ex = max(0.0, float(np.polyfit(delta, gamma, 1)[1]))    # extrapolated intercept, clipped
    return G_lb, G_ex


rows = []
for fa in sorted(set(factor.tolist())):
    for z in sorted(set(zl[factor == fa].tolist())):
        m = (factor == fa) & (zl == z)
        denom = (F[m] ** 2).sum(1).mean()
        err = ((F[m] - Fpred[m]) ** 2).sum(1).mean()
        G_lb, G_ex = gamma_test(U[m], F[m], tt[m], pos[m])
        rows.append(dict(factor=fa, zl=z, depth=float(depth[m][0]), n=int(m.sum()),
                         R2F_model=1 - err / denom,
                         R2F_ceil_lb=1 - G_lb / denom if np.isfinite(G_lb) else np.nan,
                         R2F_ceiling=1 - G_ex / denom if np.isfinite(G_ex) else np.nan))
        r = rows[-1]
        print(f'  f{fa:2d} z{z:2d} {r["depth"]:7.1f}m n={r["n"]:6d}  model={r["R2F_model"]:6.3f}  '
              f'ceil_lb={r["R2F_ceil_lb"]:6.3f}  ceiling={r["R2F_ceiling"]:6.3f}  '
              f'headroom(lb)={r["R2F_ceil_lb"]-r["R2F_model"]:+.3f}', flush=True)

print('\n=== depth-mean by resolution ===', flush=True)
import statistics as st
SPAC = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}
for scope, sel in [('all depths', lambda r: True), ('<1500 m', lambda r: r['depth'] < 1500)]:
    print(f'-- {scope} --', flush=True)
    for fa in sorted(set(factor.tolist())):
        rr = [r for r in rows if r['factor'] == fa and sel(r) and np.isfinite(r['R2F_ceiling'])]
        if not rr:
            continue
        mdl = st.mean(r['R2F_model'] for r in rr)
        lb = st.mean(r['R2F_ceil_lb'] for r in rr)
        ce = st.mean(r['R2F_ceiling'] for r in rr)
        print(f'   factor {fa:2d} (~{SPAC.get(fa,0):.1f} deg):  model {mdl:.3f}   ceil_lb {lb:.3f}   '
              f'ceiling {ce:.3f}   headroom[lb..ext] {lb-mdl:+.3f}..{ce-mdl:+.3f}', flush=True)

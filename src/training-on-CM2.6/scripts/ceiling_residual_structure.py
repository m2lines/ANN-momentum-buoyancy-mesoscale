"""Phase B2 of the predictability-ceiling study (see ../CEILING_STUDY_PLAN.md).

Does the trained ANN's residual carry STRUCTURE explainable by resolved fields the model
does NOT see? The non-dimensionalization normalizes the gradient/strain MAGNITUDES out of
the inputs (only directions/ratios survive); the kinetic energy is never an input. So if
the dimensional residual |F - F_pred|^2 correlates with rho_norm / gradv_norm / KE, those
discarded magnitudes carry recoverable signal -> input-limited. A residual uncorrelated with
every resolved probe -> at the stochastic floor.

The ABSOLUTE residual |F-Fpred|^2 scales trivially with the flux magnitude (~ s^2), which
R2F already divides out and which the model already uses (s is its scale). So we work with the
SCALE-REMOVED residual  e = |F-Fpred|^2 / s^2 = |T-P|^2  -- the model's error in the
dimensionless space it is fit in. A probe correlating with e indicates signal BEYOND the scale.
KE (velocity magnitude) is genuinely NOT in the stencil (which holds velocity GRADIENTS, not
velocity), so it is a real candidate extra input; rho_norm/gradv_norm are shown as controls
(the model collapses them into the product s, so leftover correlation would flag that the
multiplicative-scale form is lossy).

e is heavy-tailed, so a plain correlation ratio is dominated by outliers; we report tail-robust:
  - spearman rho(e, g)              : rank (monotone) association, outlier-robust
  - decile ratio                    : median(e | top-decile g) / median(e | bottom-decile g)
  - eta^2 on log e                  : variance-stabilized correlation ratio (bulk structure)

Reads the npz from dump_ceiling_table.py.  IN=table.npz.
"""
import os
import numpy as np
from scipy.stats import spearmanr

IN = os.path.expandvars(os.environ.get('IN', '/scratch/$USER/mom6/CM26_ML_models/FGR3/ceiling/table.npz'))
NBIN = int(os.environ.get('NBIN', 10))
d = np.load(IN)
e = (d['Tx'] - d['Px']) ** 2 + (d['Ty'] - d['Py']) ** 2             # scale-removed residual |T-P|^2
factor, depth = d['factor'], d['depth']
probes = {'rho_norm': d['rho_norm'], 'gradv_norm': d['gradv_norm'], 'KE': d['KE']}
print(f'{IN}: {e.shape[0]} rows', flush=True)


def eta2(e, g, nbin):
    """Correlation ratio on log e (variance-stabilized): fraction of Var(log e) explained by
    the bin-mean over quantile bins of g. Model-free bulk measure of what f(g) alone explains."""
    ok = np.isfinite(g) & np.isfinite(e) & (e > 0)
    le, g = np.log(e[ok]), g[ok]
    if len(le) < nbin * 5 or np.var(le) == 0:
        return np.nan
    edges = np.quantile(g, np.linspace(0, 1, nbin + 1)); edges[-1] += 1e-9
    b = np.clip(np.digitize(g, edges[1:-1]), 0, nbin - 1)
    grand = le.mean()
    between = sum(((le[b == k].mean() - grand) ** 2) * (b == k).sum() for k in range(nbin) if (b == k).any())
    return float(between / ((le - grand) ** 2).sum())


def decile_ratio(e, g):
    ok = np.isfinite(g) & np.isfinite(e)
    e, g = e[ok], g[ok]
    lo, hi = np.quantile(g, 0.1), np.quantile(g, 0.9)
    m_lo, m_hi = np.median(e[g <= lo]), np.median(e[g >= hi])
    return float(m_hi / (m_lo + 1e-30))


def report(tag, mask):
    e_m = e[mask]
    print(f'\n[{tag}]  n={mask.sum()}  median|F-Fpred|^2={np.median(e_m):.4g}', flush=True)
    for name, g in probes.items():
        g_m = g[mask]
        rho = spearmanr(e_m, g_m).correlation
        print(f'   {name:11s}: spearman={rho:+.3f}   decile_ratio={decile_ratio(e_m, g_m):6.2f}   '
              f'eta^2(log)={eta2(e_m, g_m, NBIN):.3f}', flush=True)


report('ALL', np.ones(len(e), bool))
for fa in sorted(set(factor.tolist())):
    report(f'factor {fa}', factor == fa)
report('depth < 500 m', depth < 500)
report('depth > 2000 m', depth > 2000)

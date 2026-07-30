"""Amplitude of the parameterized eddy impact relative to the diagnosed one, versus grid spacing.

Area-weighted global integrals of the APE sink and of the forcing, ANN over diagnosed, both for the
full column and restricted to below the mixed layer. A ratio of 1 is perfect amplitude; the point of
the figure is that the ratio is NOT flat in Delta, so no single scalar coefficient reproduces the
diagnosed energetics at every resolution.

Reads the eval_mean_impact.py output; no inference.
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

R = os.path.expandvars(os.environ.get(
    'MI', '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/mean-impact-mld-full'))
P = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d/subfilter-neutral/FGR3')
FACTORS, SPACING = [4, 9, 12, 15], [0.4, 0.9, 1.2, 1.5]

PAIRS = {'a': ('a', 'a_pred'), 'a_belowml': ('a_belowml', 'a_pred_belowml'),
         'G': ('G', 'G_pred'), 'G_belowml': ('G_belowml', 'G_pred_belowml')}
res = {k: [] for k in PAIRS}
for fa in FACTORS:
    d = xr.open_dataset(f'{R}/factor-{fa}.nc')
    pm = xr.open_dataset(f'{P}/factor-{fa}/param.nc')
    ar = np.asarray(pm.dxT * pm.dyT)
    for k, (kt, kp) in PAIRS.items():
        t, p = np.asarray(d[kt].values), np.asarray(d[kp].values)
        o = np.isfinite(t) & np.isfinite(p)
        res[k].append((p[o] * ar[o]).sum() / (t[o] * ar[o]).sum())

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for ax, (kf, kb, ttl) in zip(axes, [('a', 'a_belowml', 'APE sink'),
                                    ('G', 'G_belowml', 'forcing of the mean')]):
    ax.plot(SPACING, res[kf], 'o-', color='0.35', lw=2, label='full column')
    ax.plot(SPACING, res[kb], 's-', color='C3', lw=2, label='below mixed layer')
    ax.axhline(1.0, color='0.5', ls=':', lw=1.2)
    ax.set_xlabel(r'coarse grid spacing $\Delta$ [deg]')
    ax.set_ylabel('ANN / diagnosed  (area-weighted integral)')
    ax.set_title(ttl, fontsize=11)
    ax.set_xticks(SPACING); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.5)

fig.suptitle('Amplitude of the parameterized eddy impact vs resolution '
             '(132 snapshots, train+validate+test)', fontsize=12)
fig.savefig('mean_impact_vs_res.png', bbox_inches='tight', dpi=150)
fig.savefig('mean_impact_vs_res.pdf', bbox_inches='tight', dpi=150)

print(f'{"D":>5} ' + ' '.join(f'{k:>12}' for k in res))
for i, s in enumerate(SPACING):
    print(f'{s:>5} ' + ' '.join(f'{res[k][i]:>12.3f}' for k in res))
print('wrote mean_impact_vs_res.png/.pdf')

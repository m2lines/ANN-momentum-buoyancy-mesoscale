"""Depth profiles of the time-mean eddy impact: where in the column the scheme acts.

Area-weighted horizontal mean at each level, diagnosed vs ANN, for
  the forcing of the mean   G = -u* . grad rho
  the APE sink             a = Upsilon . grad_h rho

SIGN: positive a is a RELEASE of resolved APE. The code stores F_code = -F^rho (cm26.py:586 forms
rho_bar*u_bar - (rho*u)_bar, the negative of the paper's convention), and the appendix gives
dPE/dt = rho0 * int (F^b.grad_h b)/(d_z b) dV. Converting with F^b = -(g/rho0) F^rho and
b = -g rho/rho0, and using d_z rho = -|d_z rho| for stable stratification, gives
dPE/dt = -g * int a dV. So a > 0 <=> dPE/dt < 0 <=> APE removed from the resolved flow.

  FACTOR=9 python plot_mean_impact_profile.py
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

FACTOR = int(os.environ.get('FACTOR', 9))
SPACING = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}[FACTOR]
R = os.path.expandvars(os.environ.get(
    'PROF', '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/mean-impact-prof'))

d = xr.open_dataset(f'{R}/factor-{FACTOR}.nc')
zl = d['zl'].values
n = int(d['n_snapshots'].values)

PANELS = [('a_prof', 'a_pred_prof', r'APE sink  $\langle a\rangle$   (positive = APE released)'),
          ('G_prof', 'G_pred_prof', r'forcing of the mean  $\langle G\rangle$')]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), constrained_layout=True)

# Log depth axis: the exchange is so surface-trapped (63% of the column total sits above 200 m)
# that a linear axis spends 95% of the panel on a flat line at zero.
for ax, (kt, kp, lab) in zip(axes[:2], PANELS):
    ax.plot(d[kt].values, zl, 'k-', lw=2, label='diagnosed (CM2.6)')
    ax.plot(d[kp].values, zl, 'C3--', lw=2, label='ANN')
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_yscale('log'); ax.set_ylim(zl.max(), zl.min())
    ax.set_ylabel('depth [m]'); ax.set_title(lab, fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which='both')

# The finding: the deficit is depth-structured, so no single scalar can correct it.
ax = axes[2]
r = d['a_pred_prof'].values / np.where(np.abs(d['a_prof'].values) > 0, d['a_prof'].values, np.nan)
ax.plot(r, zl, 'C0-', lw=2)
ax.axvline(1.0, color='0.4', lw=1.0, ls=':')
ax.set_yscale('log'); ax.set_ylim(zl.max(), zl.min()); ax.set_xlim(0, 2)
ax.set_ylabel('depth [m]'); ax.set_xlabel('ANN / diagnosed')
ax.set_title('APE-sink amplitude ratio\n(1.0 = perfect; note it is not flat)', fontsize=10)
ax.grid(alpha=0.3, which='both')

fig.suptitle(f'Depth structure of the time-mean eddy impact, $\\Delta={SPACING}^\\circ$ '
             f'({n} snapshots)', fontsize=12)
fig.savefig(f'mean_impact_profile_f{FACTOR}.png', bbox_inches='tight', dpi=150)
fig.savefig(f'mean_impact_profile_f{FACTOR}.pdf', bbox_inches='tight', dpi=150)

# where does the APE exchange actually sit, and how much of it does the ANN deliver?
dz = np.gradient(zl)
ct, cp = np.cumsum(d['a_prof'].values * dz), np.cumsum(d['a_pred_prof'].values * dz)
for zc in [200, 500, 1000, 2000]:
    i = int(np.argmin(np.abs(zl - zc)))
    print(f'  above {zc:5d} m: diagnosed {100*ct[i]/ct[-1]:5.1f}% of column total, '
          f'ANN/diagnosed = {cp[i]/ct[i]:.2f}')
print(f'  column total: diagnosed {ct[-1]:.4e}  ANN {cp[-1]:.4e}  ratio {cp[-1]/ct[-1]:.2f}')
print(f'wrote mean_impact_profile_f{FACTOR}.png/.pdf')

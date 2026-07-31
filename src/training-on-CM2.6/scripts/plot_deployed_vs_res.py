"""Scaling figure: amplitude of the DEPLOYED APE release, ANN over diagnosed, vs grid spacing --
with the mixed-layer proxy drawn faintly beside it as the visible validation.

Numbers are the area-weighted global integrals of a_tap = Ups_tap . grad_h(rho) (full test split,
24 snapshots), from diag_fgnv_taper.py runs (f4/12/15: task log 2026-07-31; f9: the unmasked
integral of fgnv_forcing_maps_f9_dep.nc). The proxy values are the below-ML integrals of
eval_mean_impact.py (132 snapshots). The two agree to <=0.05 at every point, which is the proxy
validation the SI details. The forcing panel of the predecessor figure is retired: depth-averaged
forcing statistics are boundary-term-dominated (SI); deployed forcing skill is quoted in the text
from the vertical-anatomy moments instead.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt

SPACING = [0.4, 0.9, 1.2, 1.5]
DEPLOYED = [1.362, 1.139, 0.952, 0.859]     # 24-snapshot deployed-operator ratios
PROXY = [1.330, 1.105, 0.967, 0.841]        # below-ML hard cut, 132 snapshots

fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
ax.plot(SPACING, PROXY, 's--', color='0.65', lw=1.4, ms=5,
        label='mixed-layer proxy (SI)')
ax.plot(SPACING, DEPLOYED, 'o-', color='C3', lw=2.2, ms=7,
        label='deployed operator')
ax.axhline(1.0, color='0.5', ls=':', lw=1.2)
ax.set_xlabel(r'coarse grid spacing $\Delta$ [deg]')
ax.set_ylabel('APE release, ANN / diagnosed')
ax.set_xticks(SPACING)
ax.set_ylim(0.75, 1.45)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
ax.set_title('Amplitude of the deployed energy release vs resolution', fontsize=11)
fig.savefig('deployed_vs_res.png', bbox_inches='tight', dpi=150)
fig.savefig('deployed_vs_res.pdf', bbox_inches='tight', dpi=150)
print('wrote deployed_vs_res.png/.pdf')

"""Vertical section of the forcing anatomy: at what depths is the deployed (FGNV) G a faithful
representation of the diagnosed forcing, and at what depths does the vertical term dominate in
magnitude -- the distinction pattern correlation alone hides.

Three panels vs depth (log axis):
  (1) area-weighted RMS amplitude, per level, of the diagnosed forcing G, its vertical part
      -dz(a), its divergence part -div_h(F), and the deployed (tapered) G. Who wins in MAGNITUDE.
  (2) per-level pattern correlation of each part -- and of the deployed G -- with the diagnosed G.
      Who wins the PATTERN, and where deployment preserves the forcing.
  (3) the ANN's deployed-forcing skill by level: corr(pred, diag) and RMS ratio.

  FACTOR=9 SECT=... python plot_fgnv_forcing_section.py
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

FAC = int(os.environ.get('FACTOR', 9))
# YSCALE=log kept for diagnostics; linear is the presentation default now that the story is the
# interior. On a log axis the top 100 m -- the band the remit framing explicitly discounts --
# occupies half the panel and visually overstates the problem. The discounted band is SHADED and
# labelled instead, so it reads as out-of-scope rather than hidden.
YSCALE = os.environ.get('YSCALE', 'linear')
ZMAX = float(os.environ.get('ZMAX', 4000.))
BAND = 300.       # machinery-restructured band (surface crush + taper shoulder)
SPACING = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}[FAC]
SECT = os.environ.get('SECT', '') or os.path.expandvars(
    f'/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/fgnv_forcing_sect_f{FAC}.nc')

d = xr.open_dataset(SECT)
zl = d.zl.values
n = int(d.attrs['n_snapshots'])
g = lambda k: d[k].values
rms = lambda k: np.sqrt(g(k) / np.maximum(g(k + '_w'), 1.))
corr = lambda kxy, kx, ky: g(kxy) / np.sqrt(np.maximum(g(kx) * g(ky), 1e-300))

fig, ax = plt.subplots(1, 3, figsize=(13, 5.4), constrained_layout=True)

ax[0].plot(rms('gt2'), zl, 'k-', lw=2, label=r'diagnosed $G$')
ax[0].plot(rms('vt2'), zl, 'C0-', lw=1.8, label=r'vertical part $-\partial_z a$')
ax[0].plot(rms('dt2'), zl, 'C2-', lw=1.8, label=r'divergence part $-\nabla_h\!\cdot\!\mathbf{F}$')
ax[0].plot(rms('gd2'), zl, 'C3--', lw=2, label=r'DEPLOYED $G$ (FGNV)')
ax[0].set_xscale('log'); ax[0].set_yscale(YSCALE); ax[0].set_ylim(ZMAX, 0 if YSCALE=='linear' else zl.min())
ax[0].set_xlabel('area-weighted RMS'); ax[0].set_ylabel('depth [m]')
ax[0].set_title('(1) magnitude by depth', fontsize=10)
ax[0].legend(fontsize=8, loc='lower left'); ax[0].grid(alpha=0.3, which='both')

ax[1].plot(corr('gtvt', 'gt2', 'vt2'), zl, 'C0-', lw=1.8, label=r'vert part vs $G$')
ax[1].plot(corr('gtdt', 'gt2', 'dt2'), zl, 'C2-', lw=1.8, label=r'div part vs $G$')
ax[1].plot(corr('gtgd', 'gt2', 'gd2'), zl, 'C3--', lw=2, label=r'deployed $G$ vs $G$')
ax[1].axvline(0, color='0.6', lw=0.8)
ax[1].set_yscale(YSCALE); ax[1].set_ylim(ZMAX, 0 if YSCALE=='linear' else zl.min()); ax[1].set_xlim(-0.3, 1.05)
ax[1].set_xlabel('pattern correlation with diagnosed $G$'); ax[1].set_ylabel('depth [m]')
ax[1].set_title('(2) who shapes the forcing, and\nwhere deployment preserves it', fontsize=10)
ax[1].legend(fontsize=8, loc='lower left'); ax[1].grid(alpha=0.3, which='both')

ax[2].plot(corr('gdgp', 'gd2', 'gp2'), zl, 'C0-', lw=2, label='corr(pred, diag)')
ax[2].plot(np.sqrt(g('gp2') / np.maximum(g('gd2'), 1e-300)), zl, 'C1-', lw=2,
           label='RMS ratio pred/diag')
ax[2].axvline(1.0, color='0.5', ls=':'); ax[2].axvline(0, color='0.6', lw=0.8)
ax[2].set_yscale(YSCALE); ax[2].set_ylim(ZMAX, 0 if YSCALE=='linear' else zl.min()); ax[2].set_xlim(-0.2, 2.2)
ax[2].set_xlabel('deployed-forcing skill'); ax[2].set_ylabel('depth [m]')
ax[2].set_title('(3) ANN skill on the deployed\nforcing, by depth', fontsize=10)
ax[2].legend(fontsize=8, loc='lower right'); ax[2].grid(alpha=0.3, which='both')

if YSCALE == 'linear':
    for a_ in ax:
        a_.axhspan(0, BAND, color='0.85', zorder=0)
    ax[0].text(0.03, 0.985, 'restructured by the transport machinery\n(out of remit; see text)',
               transform=ax[0].get_yaxis_transform(), va='top', fontsize=7.5, color='0.35')

fig.suptitle(f'Vertical anatomy of the forcing, $\\Delta={SPACING}^\\circ$, {n} snapshots',
             fontsize=12)
sfx = '' if YSCALE == 'linear' else '_log'
fig.savefig(f'fgnv_forcing_section_f{FAC}{sfx}.png', bbox_inches='tight', dpi=150)
fig.savefig(f'fgnv_forcing_section_f{FAC}{sfx}.pdf', bbox_inches='tight', dpi=150)
print(f'wrote fgnv_forcing_section_f{FAC}{sfx}.png')

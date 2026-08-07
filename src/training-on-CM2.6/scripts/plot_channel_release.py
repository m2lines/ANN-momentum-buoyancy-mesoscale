"""Quick-look for diag_channel_release.py output: truth-diagnosed vs ANN vs fitted-constant-kappa
GM DEPLOYED APE release on the 1/4-deg channel grid (offline-on-truth vertex of the online-metrics
triangle; PLAN.md 2026-08-07 cont 3/4). Final paper figure waits for the online GMwork maps."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np, xarray as xr
import matplotlib.pyplot as plt

d = xr.open_dataset('/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/channel_release/woc_p0625_factor4.nc')
G = 9.8
m = d.mask2d == 1
ape = {k: (G * d[f'ape_{k}']).where(m) * 1e3 for k in ('diag', 'pred', 'gm')}   # mW/m^2

fig, ax = plt.subplots(2, 4, figsize=(16.5, 7), constrained_layout=True)
v = float(np.nanpercentile(np.abs(ape['diag']), 98))
panels = [(ape['diag'], 'diagnosed (truth)'), (ape['pred'], 'ANN predicted'),
          (ape['gm'], f'GM, fitted $\\kappa^*$={d.attrs["kappa_star"]:.0f} m$^2$/s'),
          (ape['pred'] - ape['diag'], 'ANN - diagnosed')]
for a_, (f, t) in zip(ax[0], panels):
    im = a_.pcolormesh(d.xh, d.yh, f, cmap='RdBu_r', vmin=-v, vmax=v)
    a_.set_title(f'g$\\int a\\,dz$ [mW m$^{{-2}}$]\n{t}', fontsize=10)
    plt.colorbar(im, ax=a_, shrink=0.8)

sect = {k: G * d[f'sect_a_{k}'] * 1e3 for k in ('diag', 'pred', 'gm')}          # mW/m^3
vs = float(np.nanpercentile(np.abs(sect['diag']), 98))
for a_, k, t in zip(ax[1], ('diag', 'pred', 'gm'),
                    ('zonal-mean release, diagnosed', 'zonal-mean release, ANN',
                     'zonal-mean release, GM($\\kappa^*$)')):
    im = a_.pcolormesh(d.yh, d.zl, sect[k], cmap='RdBu_r', vmin=-vs, vmax=vs)
    a_.invert_yaxis(); a_.set_title(t + '  [mW m$^{-3}$]', fontsize=10)
    a_.set_xlabel('lat'); a_.set_ylabel('depth (m)')
    plt.colorbar(im, ax=a_, shrink=0.8)

vu = float(np.nanpercentile(np.abs(d.sect_uy_diag), 99))
im = ax[1, 3].pcolormesh(d.yh, d.zl, d.sect_uy_diag, cmap='RdBu_r', vmin=-vu, vmax=vu)
ax[1, 3].contour(d.yh, d.zl, d.sect_uy_pred, levels=np.linspace(-vu, vu, 9),
                 colors='k', linewidths=0.6)
ax[1, 3].contour(d.yh, d.zl, d.sect_uy_gm, levels=np.linspace(-vu, vu, 9),
                 colors='g', linewidths=0.6, linestyles='--')
ax[1, 3].invert_yaxis()
ax[1, 3].set_title('tapered $\\Upsilon_y$ [m$^2$s$^{-1}$]:\ndiag (color), ANN (black), GM (green)',
                   fontsize=10)
ax[1, 3].set_xlabel('lat')
plt.colorbar(im, ax=ax[1, 3], shrink=0.8)

rp, rg = d.attrs['ratio_pred_over_diag'], d.attrs['ratio_gm_over_diag']
fig.suptitle(f'Channel woc 1/16$\\degree$ truth -> 1/4$\\degree$, deployed convention '
             f'(FGNV $\\gamma$=1), {d.attrs["nt"]} snaps; APE ratio ANN/diag = {rp:.3f}, '
             f'GM($\\kappa^*$)/diag = {rg:.3f}', fontsize=12)
fig.savefig('channel_release_quicklook.png', dpi=130, bbox_inches='tight')
print(f'wrote channel_release_quicklook.png; ANN {rp:.3f}, GM {rg:.3f}, kappa* {d.attrs["kappa_star"]:.0f}')

"""Quick-look for diag_channel_release.py output: the truth-diagnosed vs ANN-predicted DEPLOYED
APE release on the 1/4-deg channel grid (offline-on-truth vertex of the online-metrics triangle;
PLAN.md 2026-08-07 cont 3). Final paper figure waits for the online GMwork maps (Task A)."""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np, xarray as xr
import matplotlib.pyplot as plt

d = xr.open_dataset('/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/channel_release/woc_p0625_factor4.nc')
G = 9.8
m = d.mask2d == 1
ape_d = (G * d.ape_diag).where(m) * 1e3     # mW/m^2
ape_p = (G * d.ape_pred).where(m) * 1e3

fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
v = float(np.nanpercentile(np.abs(ape_d), 98))
for a_, f, t in [(ax[0, 0], ape_d, 'diagnosed (truth)'), (ax[0, 1], ape_p, 'ANN predicted'),
                 (ax[0, 2], ape_p - ape_d, 'ANN - diagnosed')]:
    im = a_.pcolormesh(d.xh, d.yh, f, cmap='RdBu_r', vmin=-v, vmax=v)
    a_.set_title(f'deployed APE release  g$\\int a\\,dz$ [mW m$^{{-2}}$]\n{t}', fontsize=10)
    plt.colorbar(im, ax=a_, shrink=0.8)

sa_d, sa_p = G * d.sect_a_diag * 1e3, G * d.sect_a_pred * 1e3   # mW/m^3 (per m depth)
vs = float(np.nanpercentile(np.abs(sa_d), 98))
for a_, f, t in [(ax[1, 0], sa_d, 'zonal-mean release, diagnosed'),
                 (ax[1, 1], sa_p, 'zonal-mean release, ANN')]:
    im = a_.pcolormesh(d.yh, d.zl, f, cmap='RdBu_r', vmin=-vs, vmax=vs)
    a_.invert_yaxis(); a_.set_title(t + '  [mW m$^{-3}$]', fontsize=10)
    a_.set_xlabel('lat'); a_.set_ylabel('depth (m)')
    plt.colorbar(im, ax=a_, shrink=0.8)

vu = float(np.nanpercentile(np.abs(d.sect_uy_diag), 99))
im = ax[1, 2].pcolormesh(d.yh, d.zl, d.sect_uy_diag, cmap='RdBu_r', vmin=-vu, vmax=vu)
cs = ax[1, 2].contour(d.yh, d.zl, d.sect_uy_pred, levels=np.linspace(-vu, vu, 9),
                      colors='k', linewidths=0.6)
ax[1, 2].invert_yaxis()
ax[1, 2].set_title('tapered $\\Upsilon_y$ [m$^2$s$^{-1}$]: diag (color), ANN (contours)', fontsize=10)
ax[1, 2].set_xlabel('lat')
plt.colorbar(im, ax=ax[1, 2], shrink=0.8)

r = d.attrs['ratio_pred_over_diag']
fig.suptitle(f'Channel woc 1/16$\\degree$ truth -> 1/4$\\degree$, deployed convention '
             f'(FGNV $\\gamma$=1), {d.attrs["nt"]} snaps; APE ratio ANN/diag = {r:.3f}', fontsize=12)
fig.savefig('channel_release_quicklook.png', dpi=130, bbox_inches='tight')
print('wrote channel_release_quicklook.png; ratio', r)

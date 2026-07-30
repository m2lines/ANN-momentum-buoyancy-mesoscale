"""Plot the time-mean eddy impact maps written by eval_mean_impact.py.

Two rows: the forcing of the mean flow (depth average of G = -u*.grad rho) and the APE sink
(depth integral of a = Upsilon.grad_h rho). Three columns: diagnosed from CM2.6, predicted by the
ANN, and their difference. No skill score -- these are property plots.

  FACTOR=9  python plot_mean_impact.py
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from helpers.plot_helpers import regrid_tripolar

FACTOR = int(os.environ.get('FACTOR', 9))
SPACING = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}[FACTOR]
R = os.path.expandvars(os.environ.get(
    'MI', '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/mean-impact-mld'))
P = os.path.expandvars(f'/scratch/$USER/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-{FACTOR}')

d = xr.open_dataset(f'{R}/factor-{FACTOR}.nc')
pm = xr.open_dataset(f'{P}/param.nc')
glon, glat = pm['geolon'].values, pm['geolat'].values
n = int(d['n_snapshots'].values)

# Per-row colour percentile. G is far more heavy-tailed than a -- its 97th percentile is 4.5x its
# 90th -- so a common limit buries the open-ocean structure inside the innermost fifth of the scale
# and the row reads as washed out. Clip G harder and let the coastal extremes saturate (the
# colorbar is extended so the saturation is declared rather than hidden).
# The APE row is integrated only BELOW the mixed layer. Online the streamfunction is tapered toward
# the surface (Ferrari/FGNV) and mixed-layer restratification belongs to a separate submesoscale
# closure, so the ML part of the column is not what this scheme is asked to deliver: it holds 58% of
# the diagnosed column-integrated release and including it flips the apparent bias (0.92 full column
# vs 1.11 below the ML).
ROWS = [('G', 'G_pred', r'eddy forcing of the mean, $\langle G\rangle$ (depth avg)', 90),
        ('a_belowml', 'a_pred_belowml',
         r'APE sink below the mixed layer, $\int_{\rm ML}\!\langle a\rangle\,dz$', 97)]

bal = cmocean.cm.balance.copy(); bal.set_bad('white')
proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(15, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

for r, (kt, kp, label, pct) in enumerate(ROWS):
    t = np.asarray(d[kt].values, dtype='float64')
    p = np.asarray(d[kp].values, dtype='float64')
    loni, lati, tr = regrid_tripolar(t, glon, glat, SPACING)
    _, _, pr = regrid_tripolar(p, glon, glat, SPACING)
    dr = pr - tr
    # sign check: is the net APE contribution a release, as the down-gradient limit requires?
    print(f'{kt}: area-mean diagnosed={np.nanmean(t):.3e}  ANN={np.nanmean(p):.3e}  '
          f'ratio={np.nanmean(p)/np.nanmean(t):.2f}', flush=True)

    vmax = np.nanpercentile(np.abs(tr), pct)
    axs = []
    for c, (arr, ttl) in enumerate([(tr, 'diagnosed (CM2.6)'), (pr, 'ANN'), (dr, 'ANN $-$ CM2.6')]):
        ax = fig.add_subplot(gs[r, c], projection=proj); axs.append(ax)
        im = ax.pcolormesh(loni, lati, arr, cmap=bal, vmin=-vmax, vmax=vmax, zorder=1,
                           transform=ccrs.PlateCarree(), rasterized=True)
        ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
        ax.coastlines(lw=0.3, zorder=4); ax.set_global()
        if r == 0:
            ax.set_title(ttl, fontsize=11)
        if c == 0:
            ax.text(-0.04, 0.5, label, transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=10)
    fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.8, pad=0.02, extend='both')

# median over WET columns only: land carries the "no MLD found" fallback (the deepest level), which
# would otherwise drag the quoted median up by ~30 m
_wet = np.asarray(pm.wet.isel(zl=0).values) > 0.5 if 'zl' in pm.wet.dims else np.asarray(pm.wet.values) > 0.5
_m = np.asarray(d['mld'].values)
mldmed = float(np.median(_m[_wet & np.isfinite(_m) & (_m < 5000)]))
fig.suptitle(f'Time-mean eddy impact, $\\Delta={SPACING}^\\circ$ ({n} snapshots); '
             f'APE integrated below the mixed layer', fontsize=13)
fig.savefig(f'mean_impact_f{FACTOR}.png', bbox_inches='tight', dpi=150)
fig.savefig(f'mean_impact_f{FACTOR}.pdf', bbox_inches='tight', dpi=150)
print(f'wrote mean_impact_f{FACTOR}.png/.pdf from {n} snapshots')


# --- mixed-layer depth used for the cut, as context for what was removed -------------------------
figm = plt.figure(figsize=(7.0, 3.8), constrained_layout=True)
axm = figm.add_subplot(1, 1, 1, projection=proj)
_, _, mr = regrid_tripolar(np.asarray(d['mld'].values, dtype='float64'), glon, glat, SPACING)
imm = axm.pcolormesh(loni, lati, mr, cmap=cmocean.cm.deep, vmin=0, vmax=200,
                     transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
axm.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
axm.coastlines(lw=0.3, zorder=4); axm.set_global()
figm.colorbar(imm, ax=axm, orientation='vertical', shrink=0.85, pad=0.02, extend='max',
              label='mixed-layer depth [m]')
axm.set_title(f'Mixed-layer depth used for the cut (median {mldmed:.0f} m, '
              f'$\\Delta\\sigma_0=0.03$)', fontsize=10)
figm.savefig(f'mean_impact_mld_f{FACTOR}.png', bbox_inches='tight', dpi=150)
figm.savefig(f'mean_impact_mld_f{FACTOR}.pdf', bbox_inches='tight', dpi=150)
print(f'wrote mean_impact_mld_f{FACTOR}.png/.pdf  (median MLD {mldmed:.0f} m)')

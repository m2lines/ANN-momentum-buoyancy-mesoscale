"""Explanatory figure: what went wrong with the forcing metric, and what the deployed forcing
actually looks like.

Top row -- the anatomy of the OLD metric (clamp15, diagnosed fields): the depth-averaged forcing
G splits into a vertical part, -dz(a), and a horizontal-divergence part, -div_h(F). The maps show
that the depth-averaged G is essentially the vertical part in disguise -- and the vertical part's
depth average is just the surface APE-sink density leaking through the average, because
depth-averaging a vertical derivative leaves the boundary values. So the 'forcing amplitude
ratio' quoted so far was largely re-measuring the surface energetics.

Bottom row -- the DEPLOYED forcing (FGNV taper applied, which drives a -> 0 at both boundaries and
thereby deletes the vertical term's depth average): diagnosed vs predicted vs difference. What is
left is the divergence-type field, the network's weakest quantity: pattern corr ~0.4, amplitude
~1.5.

  FACTOR=9 python plot_fgnv_forcing_problem.py
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

FAC = int(os.environ.get('FACTOR', 9))
SPACING = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}[FAC]
M = os.environ.get('MAPS', '') or os.path.expandvars(
    f'/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/fgnv_forcing_maps_f{FAC}.nc')
SUFFIX = os.environ.get('SUFFIX', '')
P = os.path.expandvars(f'/scratch/$USER/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-{FAC}')

d = xr.open_dataset(M)
pm = xr.open_dataset(f'{P}/param.nc')
glon, glat = pm['geolon'].values, pm['geolat'].values
area = np.asarray(pm.dxT * pm.dyT)
n = int(d.attrs['n_snapshots'])

def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    corr = np.sum(a[ok] * b[ok] * area[ok]) / np.sqrt(
        np.sum(a[ok] ** 2 * area[ok]) * np.sum(b[ok] ** 2 * area[ok]))
    return corr

def am(a):
    ok = np.isfinite(a)
    return np.sum(a[ok] * area[ok]) / np.sum(area[ok])

bal = cmocean.cm.balance.copy(); bal.set_bad('white')
proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(15, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

TOP = [('c15_total', r'(a) depth-avg $G$ (old metric), diagnosed'),
       ('c15_vert', r'(b) vertical part only, $-\overline{\partial_z a}$'),
       ('c15_div', r'(c) divergence part only, $-\overline{\nabla_h\!\cdot\!\mathbf{F}}$')]
BOT = [('dep_diag', r'(d) DEPLOYED $G$ (after FGNV), diagnosed'),
       ('dep_pred', r'(e) deployed $G$, ANN'),
       (None, r'(f) difference, ANN $-$ diagnosed')]

tot = np.asarray(d['c15_total'].values)
vmax1 = np.nanpercentile(np.abs(tot), 92)
axs = []
for c, (k, ttl) in enumerate(TOP):
    f_ = np.asarray(d[k].values)
    ax = fig.add_subplot(gs[0, c], projection=proj); axs.append(ax)
    loni, lati, fr = regrid_tripolar(f_, glon, glat, SPACING)
    ax.pcolormesh(loni, lati, fr, cmap=bal, vmin=-vmax1, vmax=vmax1,
                  transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
    ax.coastlines(lw=0.3, zorder=4); ax.set_global()
    extra = ''
    if k == 'c15_vert':
        extra = f'  [corr with (a): {stats(f_, tot):.2f}]'
    if k == 'c15_div':
        extra = f'  [corr with (a): {stats(f_, tot):.2f}]'
    ax.set_title(f'{ttl}\n' + r'$\langle\cdot\rangle$ = ' + f'{am(f_):.1e}' + extra, fontsize=9)

dd_, dp_ = np.asarray(d['dep_diag'].values), np.asarray(d['dep_pred'].values)
vmax2 = np.nanpercentile(np.abs(dd_), 92)
for c, (k, ttl) in enumerate(BOT):
    f_ = dp_ - dd_ if k is None else np.asarray(d[k].values)
    ax = fig.add_subplot(gs[1, c], projection=proj); axs.append(ax)
    loni, lati, fr = regrid_tripolar(f_, glon, glat, SPACING)
    ax.pcolormesh(loni, lati, fr, cmap=bal, vmin=-vmax2, vmax=vmax2,
                  transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
    ax.coastlines(lw=0.3, zorder=4); ax.set_global()
    extra = f'  [corr diag: {stats(dp_, dd_):.2f}]' if k == 'dep_pred' else ''
    ax.set_title(f'{ttl}\n' + r'$\langle\cdot\rangle$ = ' + f'{am(f_):.1e}' + extra, fontsize=9)

fig.suptitle(('BELOW-ML averages. ' if SUFFIX else '') + f'Why the forcing metric misled us, $\\Delta={SPACING}^\\circ$, {n}-snapshot time '
             f'mean.  Top: the old depth-averaged $G$ (a) is dominated by its vertical part (b) '
             f'--- i.e.\\ surface energetics --- not the divergence part (c).\n'
             f'Bottom: the deployed (FGNV-tapered) forcing deletes (b) by construction; what '
             f'remains is divergence-like, and there the ANN is noisy.', fontsize=10)
fig.savefig(f'fgnv_forcing_problem_f{FAC}{SUFFIX}.png', bbox_inches='tight', dpi=150)
fig.savefig(f'fgnv_forcing_problem_f{FAC}{SUFFIX}.pdf', bbox_inches='tight', dpi=150)
print(f'wrote fgnv_forcing_problem_f{FAC}{SUFFIX}.png')

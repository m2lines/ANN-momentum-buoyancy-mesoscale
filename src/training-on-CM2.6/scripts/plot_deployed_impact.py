"""Main-text figure: the time-mean eddy impact AS DEPLOYED -- both quantities after the actual
MOM6 operator chain (unclamped Upsilon -> FGNV boundary-value solve), nothing proxied.

Top row: deployed APE release, depth-integrated sum(a_tap*dz). With Upsilon tapered, the release
is interior by construction -- the operator itself performs the separation the mixed-layer cut
proxied. Bottom row: deployed forcing, averaged over the interior band 300-3000 m (the depth-mean
of the deployed G is boundary-term-free but mixes the machinery's transition band into the
picture; the band average shows the region the anatomy section demonstrates is faithful).
Both rows masked >=2 cells from topography per level and >=2 levels above the local seafloor.

  FACTOR=9 python plot_deployed_impact.py
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
M = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/'
                       f'fgnv_forcing_maps_f{FAC}_dep.nc')
P = os.path.expandvars(f'/scratch/$USER/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-{FAC}')

d = xr.open_dataset(M)
pm = xr.open_dataset(f'{P}/param.nc')
glon, glat = pm['geolon'].values, pm['geolat'].values
area = np.asarray(pm.dxT * pm.dyT)
n = int(d.attrs['n_snapshots'])

ape_t = np.asarray(d['dep_ape_diag'].values)
ape_p = np.asarray(d['dep_ape_pred'].values)
nb = np.asarray(d['dep_band_n'].values)
frc_t = np.where(nb > 0, np.asarray(d['dep_band_diag'].values) / np.where(nb > 0, nb, 1), np.nan)
frc_p = np.where(nb > 0, np.asarray(d['dep_band_pred'].values) / np.where(nb > 0, nb, 1), np.nan)

def wsum(a):
    ok = np.isfinite(a); return np.sum(a[ok] * area[ok])
def pstats(t, p):
    ok = np.isfinite(t) & np.isfinite(p)
    dd = np.sum(t[ok]**2 * area[ok]); pp = np.sum(p[ok]**2 * area[ok])
    pd = np.sum(t[ok] * p[ok] * area[ok])
    return pd / dd, pd / np.sqrt(dd * pp), np.sqrt(pp / dd)

ape_ratio = wsum(ape_p) / wsum(ape_t)
fs, fc, fa = pstats(frc_t, frc_p)
print(f'deployed APE integral ratio pred/diag: {ape_ratio:.3f}')
print(f'deployed band forcing: slope {fs:.3f}  corr {fc:.3f}  amp {fa:.3f}')

bal = cmocean.cm.balance.copy(); bal.set_bad('white')
proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(15, 6.4), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

ROWS = [(ape_t, ape_p, r'APE release, $\int a\,\mathrm{d}z$ (deployed)', 97),
        (frc_t, frc_p, r'forcing $G$, 300--3000 m (deployed)', 95)]
for r, (t_, p_, lab, pct) in enumerate(ROWS):
    vmax = np.nanpercentile(np.abs(t_), pct)
    axs = []
    for c, (arr, ttl) in enumerate([(t_, 'diagnosed (CM2.6)'), (p_, 'ANN'),
                                    (p_ - t_, 'ANN $-$ CM2.6')]):
        ax = fig.add_subplot(gs[r, c], projection=proj); axs.append(ax)
        loni, lati, fr = regrid_tripolar(arr, glon, glat, SPACING)
        im = ax.pcolormesh(loni, lati, fr, cmap=bal, vmin=-vmax, vmax=vmax,
                           transform=ccrs.PlateCarree(), rasterized=True, zorder=1)
        ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
        ax.coastlines(lw=0.3, zorder=4); ax.set_global()
        if r == 0:
            ax.set_title(ttl, fontsize=11)
        if c == 0:
            ax.text(-0.04, 0.5, lab, transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=10)
    fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.8, pad=0.02, extend='both')

fig.suptitle(f'Time-mean eddy impact as deployed (after the transport machinery), '
             f'$\\Delta={SPACING}^\\circ$, {n} snapshots.  '
             f'APE ratio {ape_ratio:.2f}; forcing corr {fc:.2f}, amp {fa:.2f}', fontsize=12)
fig.savefig(f'deployed_impact_f{FAC}.png', bbox_inches='tight', dpi=150)
fig.savefig(f'deployed_impact_f{FAC}.pdf', bbox_inches='tight', dpi=150)
print(f'wrote deployed_impact_f{FAC}.png/.pdf')

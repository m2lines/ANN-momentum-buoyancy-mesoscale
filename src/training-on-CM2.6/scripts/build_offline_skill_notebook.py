"""Build the offline-skill paper-figure notebook (sec:offline) and render its PDF.

Writes DB_notebooks/paper_figure_notebooks/offline_skill.ipynb (plain .ipynb JSON,
Pavel_Container kernel) -- the reproducible figure source in the style of Pavel's
notebooks/Figure-1.ipynb -- and runs the cell code headless to drop offline_skill.pdf
(the container lacks nbformat, so we emit JSON + exec). Run in the Pavel container.

Figure: (a) global Robinson maps of div(F^rho) diagnosed vs ANN; (b) Gulf-Stream zoom of the
along/across-gradient flux (diagnosed vs ANN); (c) GROUPED heatmaps -- each coarse-grid spacing
is a box of three sub-bins (along | across | div skill) over depth, for R^2 and correlation.
"""
import os
import json
os.environ['MPLBACKEND'] = 'Agg'

NB_DIR = '/home/db194/ANN-momentum-buoyancy-mesoscale/DB_notebooks/paper_figure_notebooks'
os.makedirs(NB_DIR, exist_ok=True)

SETUP = """import os, sys
sys.path.append('../../src/training-on-CM2.6')   # for helpers.cm26.create_grid
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from helpers.cm26 import create_grid

# fp64 test-skill files WITH the divergence metric (skill-test-rho-div: adds R2F_div/corr_F_div
# to the along/across/combined skills; also stores the snapshot fields for the maps).
ROOT  = os.path.expandvars(os.environ.get('SKILL_ROOT', '/scratch/$USER/CM26_ML_models/FGR3/EXP0/skill-test-rho-div'))
PARAM = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d/subfilter/FGR3')
FACTORS = [4, 9, 12, 15]; SPACING = [0.4, 0.9, 1.2, 1.5]
MAP_FA = 9; MAP_DEPTH = 500.0; GS_BOX = (-80, -45, 30, 48)   # Gulf Stream lon/lon/lat/lat
METRICS = ['along', 'across', 'div']; ABBR = {'along': 'al', 'across': 'ac', 'div': 'dv'}"""

LOAD = """sk = {fa: xr.open_dataset(f'{ROOT}/factor-{fa}.nc') for fa in FACTORS}
zl = sk[FACTORS[0]]['zl'].values
# skill[metric] = (factor, depth); missing metric (e.g. div before recompute) -> NaN placeholder
def getm(fa, name):
    return sk[fa][name].values if name in sk[fa] else np.full(len(zl), np.nan)
# coast-excluded (>=2 cells from coast, per-depth) variants -- matches Perezhogin et al. convention
R2 = {m: np.array([getm(fa, f'R2F_{m}_away') for fa in FACTORS]) for m in METRICS}
CO = {m: np.array([getm(fa, f'corr_F_{m}_away') for fa in FACTORS]) for m in METRICS}
d = sk[MAP_FA]; pm = xr.open_dataset(f'{PARAM}/factor-{MAP_FA}/param.nc'); grid = create_grid(pm)
lon, lat = pm['geolon'].values, pm['geolat'].values
k = int(np.argmin(np.abs(zl - MAP_DEPTH)))
wet2d = (pm.wet.isel(zl=k) if 'zl' in pm.wet.dims else pm.wet)"""

DIV = """# metric-correct C-grid flux-form divergence of a T-point horizontal flux (xgcm grid)
def divergence(Fx, Fy):
    fx = xr.DataArray(Fx.astype('float64'), dims=['yh', 'xh'], coords={'yh': pm.yh, 'xh': pm.xh}).fillna(0.)
    fy = xr.DataArray(Fy.astype('float64'), dims=['yh', 'xh'], coords={'yh': pm.yh, 'xh': pm.xh}).fillna(0.)
    div = (grid.diff(grid.interp(fx, 'X') * pm.dyCu, 'X')
           + grid.diff(grid.interp(fy, 'Y') * pm.dxCv, 'Y')) / (pm.dxT * pm.dyT)
    return xr.where(wet2d > 0.5, div, np.nan).values

divT = divergence(d['Fx'].isel(zl=k).values, d['Fy'].isel(zl=k).values)
divP = divergence(d['Fx_pred'].isel(zl=k).values, d['Fy_pred'].isel(zl=k).values)

# The CM2.6 grid is TRIPOLAR (its geolon is discontinuous near the Arctic), which breaks cartopy
# pcolormesh and blanks the NH. Regrid the curvilinear field to a regular 1-deg lon/lat grid for
# plotting -- robust, and the cartopy LAND feature (drawn on top) hides interpolation over land.
from scipy.interpolate import griddata
lon2 = ((lon + 180) % 360) - 180
loni = np.arange(-179.5, 180., 1.0); lati = np.arange(-78.5, 89.5, 1.0)
LO, LA = np.meshgrid(loni, lati)
def regrid(f):
    ok = np.isfinite(f)
    pts, val = np.column_stack([lon2[ok], lat[ok]]), f[ok]
    # pad periodically in lon so the interpolation wraps cleanly across +/-180
    pts = np.vstack([pts, pts + [360, 0], pts - [360, 0]]); val = np.concatenate([val, val, val])
    lin = griddata(pts, val, (LO, LA), method='linear')
    nn = griddata(pts, val, (LO, LA), method='nearest')   # fill convex-hull gaps
    return np.where(np.isfinite(lin), lin, nn)
divT_r, divP_r = regrid(divT), regrid(divP)"""

PLOT = """bal = cmocean.cm.balance.copy(); bal.set_bad('white')   # data gaps (shelves below seafloor) -> white
plt.rcParams.update({'font.size': 11})
fig = plt.figure(figsize=(13.5, 8.8), constrained_layout=True)
outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 0.9])   # shorter heatmap row (room for labels)
top = outer[0].subgridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.10)
gmap = top[0].subgridspec(2, 1, hspace=0.12)        # global maps STACKED (diagnosed over ANN)
gbot = top[1].subgridspec(2, 2, wspace=0.06, hspace=0.15)
bot = outer[1].subgridspec(1, 2, wspace=0.22)
proj = ccrs.Robinson(central_longitude=0)   # matches regridded -180..180 (no wrap -> full render)

def land(ax):     # clean grey continents on top of the data; coastline outline
    ax.add_feature(cfeature.LAND, facecolor='0.8', edgecolor='none', zorder=3)
    ax.coastlines(lw=0.3, zorder=4)

# (a) global divergence maps STACKED: diagnosed (top) / ANN (bottom).
# Use a low (85th-pct) colour limit so the faint open-ocean divergence shows colour rather than
# washing out to white -- |div| is intermittent (strong only in jets), so a high limit leaves
# most of the ocean near-white.
vmax = np.nanpercentile(np.abs(divT_r), 93); axs = []   # relaxed range (less saturated)
for r, (arr, ttl) in enumerate([(divT_r, 'diagnosed (CM2.6)'), (divP_r, 'ANN')]):
    ax = fig.add_subplot(gmap[r], projection=proj); axs.append(ax)
    im = ax.pcolormesh(loni, lati, arr, cmap=bal, vmin=-vmax, vmax=vmax, zorder=1,
                       transform=ccrs.PlateCarree(), rasterized=True)
    land(ax); ax.set_global(); ax.set_title(ttl, fontsize=11)
    ax.plot([GS_BOX[0], GS_BOX[1], GS_BOX[1], GS_BOX[0], GS_BOX[0]],
            [GS_BOX[2], GS_BOX[2], GS_BOX[3], GS_BOX[3], GS_BOX[2]], 'k-', lw=0.8, zorder=5,
            transform=ccrs.PlateCarree())
fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.85, pad=0.02,
             label=r'$\\nabla_h\\!\\cdot\\!\\mathbf{F}^\\rho_h$ at %.0f m, $\\Delta=%.1f^\\circ$' % (zl[k], SPACING[FACTORS.index(MAP_FA)]))

# (b) Gulf-Stream zoom: along / across, diagnosed | ANN. Regrid the curvilinear box to a
# regular grid (cartopy can't draw the native grid) then plot on PlateCarree with coastlines.
bl = np.arange(GS_BOX[0], GS_BOX[1] + .01, 0.4); ba = np.arange(GS_BOX[2], GS_BOX[3] + .01, 0.4)
BLO, BLA = np.meshgrid(bl, ba)
near = (lon >= GS_BOX[0] - 3) & (lon <= GS_BOX[1] + 3) & (lat >= GS_BOX[2] - 3) & (lat <= GS_BOX[3] + 3)
def regrid_box(f):
    ok = near & np.isfinite(f)
    return griddata(np.column_stack([lon[ok], lat[ok]]), f[ok], (BLO, BLA), method='linear')
for r, comp in enumerate(['along', 'across']):
    tg = regrid_box(d[f'F_{comp}'].isel(zl=k).values); pg = regrid_box(d[f'F_{comp}_pred'].isel(zl=k).values)
    vmx = float(np.nanpercentile(np.abs(tg), 96)); row_axes = []
    for c, (arr, lab) in enumerate([(tg, 'CM2.6'), (pg, 'ANN')]):
        ax = fig.add_subplot(gbot[r, c], projection=ccrs.PlateCarree()); row_axes.append(ax)
        im2 = ax.pcolormesh(bl, ba, arr, cmap=bal, vmin=-vmx, vmax=vmx, zorder=1,
                            transform=ccrs.PlateCarree(), rasterized=True)
        ax.set_extent([GS_BOX[0], GS_BOX[1], GS_BOX[2], GS_BOX[3]], ccrs.PlateCarree()); land(ax)
        if r == 0:
            ax.set_title(lab, fontsize=10)
        if c == 0:
            ax.text(-0.13, 0.5, r'$F^\\rho$ %s-$\\nabla\\rho$' % comp, transform=ax.transAxes,
                    rotation=90, va='center', fontsize=10)
        if r == 0 and c == 0:
            ax.text(0.03, 0.06, 'Gulf Stream', transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(fc='w', ec='none', alpha=0.7))
    fig.colorbar(im2, ax=row_axes, fraction=0.046, pad=0.02)

# (c) grouped heatmaps along the BOTTOM (wide): per scale a box of 3 sub-bins along|across|div
ze = np.concatenate([[0], 0.5 * (zl[1:] + zl[:-1]), [zl[-1] + (zl[-1] - zl[-2]) / 2]])
GAP = 0.8
def grouped(ax, M, label, vmin, cmap):
    xc, tlab, centers = [], [], []
    x = 0.0
    for i, fa in enumerate(FACTORS):
        start = x
        for m in METRICS:
            ax.pcolormesh([x - 0.5, x + 0.5], ze, M[m][i][:, None], cmap=cmap,
                          vmin=vmin, vmax=1.0, rasterized=True)
            xc.append(x); tlab.append(ABBR[m]); x += 1
        centers.append((start + x - 1) / 2); x += GAP
        if i < len(FACTORS) - 1:
            ax.axvline(x - GAP / 2, color='k', lw=0.6)
    ax.set_xlim(-0.7, x - GAP + 0.7); ax.set_ylim(ze[-1], ze[0])
    ax.set_xticks(xc); ax.set_xticklabels(tlab, fontsize=8)              # al/ac/dv along the bottom
    ax.set_xlabel(r'al = along-$\\nabla\\rho$,   ac = across-$\\nabla\\rho$,   dv = horiz. divergence', fontsize=9)
    ax.set_ylabel('depth [m]')
    sec = ax.secondary_xaxis('top')                                      # coarse-grid spacing on top
    sec.set_xticks(centers); sec.set_xticklabels([f'{s:g}$^\\circ$' for s in SPACING], fontsize=10)
    sec.tick_params(length=0)
    ax.set_title(label, pad=22)                                          # above the spacing labels
    return ax.collections[0]

for j, (M, label, vmin, cmap) in enumerate([(R2, r'$R^2$', 0.0, 'Reds'), (CO, 'correlation', 0.4, 'viridis')]):
    ax = fig.add_subplot(bot[j])
    im = grouped(ax, M, label, vmin, cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

fig.suptitle('Offline skill of the sub-grid buoyancy-flux network (held-out test split)', fontsize=13)
fig.savefig('offline_skill.pdf', bbox_inches='tight', dpi=150)
fig.savefig('offline_skill.png', bbox_inches='tight', dpi=150)
print('rendered; depth-mean R2F_div:', {SPACING[i]: round(float(np.nanmean(R2['div'][i])), 3) for i in range(4)})"""

CELLS = [
    ('md', "# Offline-skill figure (Section~3.3)\n\n"
           "Composite, in the style of Perezhogin et al. (2025) Fig. 1 but for the buoyancy "
           "problem, on the held-out test split:\n"
           "- global Robinson maps of the **horizontal divergence** of the horizontal sub-grid "
           "buoyancy flux $\\nabla_h\\cdot\\mathbf{F}^\\rho_h$ (diagnosed vs ANN) at 500 m;\n"
           "- a **Gulf Stream** zoom of the along-/across-gradient flux (diagnosed vs ANN);\n"
           "- **grouped heatmaps**: each coarse-grid spacing is a box of three sub-bins "
           "(along | across | div skill) over depth, for $R^2$ and correlation.\n\n"
           "Run top-to-bottom with the Pavel_Container kernel; copy `offline_skill.pdf` into "
           "the paper repo `figures/`."),
    ('code', SETUP), ('code', LOAD), ('code', DIV), ('code', PLOT),
]


def cell(t, src):
    lines = src.splitlines(keepends=True)
    if t == 'md':
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


nb = {"cells": [cell(t, s) for t, s in CELLS],
      "metadata": {"kernelspec": {"display_name": "Pavel_Container", "language": "python", "name": "Pavel_Container"}},
      "nbformat": 4, "nbformat_minor": 5}
with open(f'{NB_DIR}/offline_skill.ipynb', 'w') as fh:
    json.dump(nb, fh, indent=1)
print('built offline_skill.ipynb', flush=True)

os.chdir(NB_DIR)
ns = {}
for t, src in CELLS:
    if t == 'code':
        exec(src, ns)
print('rendered offline_skill.pdf OK', flush=True)

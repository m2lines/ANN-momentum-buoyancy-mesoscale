"""Build the offline-skill paper-figure notebook (sec:offline) and render its PDF.

Writes DB_notebooks/paper_figure_notebooks/offline_skill.ipynb (plain .ipynb JSON,
Pavel_Container kernel) -- the reproducible figure source in the style of Pavel's
notebooks/Figure-1.ipynb -- and runs the cell code headless to drop offline_skill.pdf
(the container lacks nbformat, so we emit JSON + exec). Run in the Pavel container.

Figure: (a) global Robinson maps of the applied forcing G = -u*.grad rho, diagnosed vs ANN;
(b) Gulf-Stream zoom of the along/across-gradient flux (diagnosed vs ANN); (c) GROUPED heatmaps --
each coarse-grid spacing is a box of four sub-bins (along | across | forcing | APE release) over
depth, for R^2 and correlation. div_h F is deliberately not shown: it is not what the model feels.
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
from helpers.cm26 import create_grid, propagate_mask

# fp64 test-skill files from skill-test-forceape: these carry the two metrics the scheme actually
# applies -- the forcing G = -u*.grad rho and the APE-sink a = Upsilon.grad_h rho -- alongside the
# flux skills, plus the snapshot fields (incl. the G/G_pred maps) the figure draws.
ROOT  = os.path.expandvars(os.environ.get('SKILL_ROOT', '/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/skill-test-forceape'))
PARAM = os.path.expandvars('/scratch/$USER/CM26_datasets/ocean3d/subfilter-neutral/FGR3')
FACTORS = [4, 9, 12, 15]; SPACING = [0.4, 0.9, 1.2, 1.5]
MAP_FA = 9; MAP_DEPTH = 500.0; GS_BOX = (-80, -45, 30, 48)   # Gulf Stream lon/lon/lat/lat
# div_h F is deliberately NOT shown: the Ferrari division by |grad_3 rho| precedes the derivative,
# so it is not the quantity the model feels (see the flux-decomposition appendix).
METRICS = ['along', 'across', 'force', 'ape']
ABBR = {'along': 'al', 'across': 'ac', 'force': 'fc', 'ape': 'pe'}"""

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
# Mask the forcing maps >=2 cells from the coast (same per-depth wet2 as the skill metric):
# G carries horizontal and vertical derivatives, so it is contaminated within 1 cell of the coast
# by the fillna(0) land BC.
_wlev = pm.wet.isel(zl=k) if 'zl' in pm.wet.dims else pm.wet
wet2d = xr.where(propagate_mask(_wlev, grid, niter=2) < 0.5, np.nan, 1.)"""

FORCE = """# The forcing the scheme applies, G = -u*.grad rho, taken straight from the snapshot that
# SGS_skill_rho stores: Upsilon = F_h/|grad_3 rho| with the MOM_meso_sfn_ANN.F90 limiters, then the
# flux form G = -div_h F_h - d/dz(Upsilon.grad_h rho). Reading it rather than rebuilding it here
# guarantees the map and the R^2 heatmap come from the same operator.
wet_np = np.asarray(wet2d)
def masked(f):
    return np.where(wet_np > 0.5, f, np.nan)      # NaN in wet2d compares False -> masked

GT = masked(d['G'].isel(zl=k).values)
GP = masked(d['G_pred'].isel(zl=k).values)

# The CM2.6 grid is TRIPOLAR (its geolon is discontinuous near the Arctic), which breaks cartopy
# pcolormesh and blanks the NH. Regrid the curvilinear field to a regular 1-deg lon/lat grid.
#
# The nearest-neighbour fill below patches the gaps the linear interpolation leaves, but on its own
# it FABRICATES data: interior holes -- shallow and marginal seas that lie below the seafloor at this
# level, and the 2-cell coastal exclusion band -- get filled by extrapolating from whatever wet point
# happens to be nearest, and they are not hidden by the cartopy LAND feature the way continents are.
# Measured on the R^2 maps, that put fabricated values in >50% of the cells drawn at the low end of
# the scale, in exactly the shallow regions. So blank anything farther than 1.5 source cells from a
# real point. Distance is taken with longitude scaled by cos(lat), otherwise the threshold is far too
# permissive at high latitude.
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
loni = np.arange(-179.5, 180., 1.0); lati = np.arange(-78.5, 89.5, 1.0)
LO, LA = np.meshgrid(loni, lati)
QXY = np.column_stack([LO.ravel() * np.cos(np.deg2rad(LA.ravel())), LA.ravel()])

def regrid_ll(f, glon, glat, dx_deg):
    lo = ((glon + 180) % 360) - 180
    ok = np.isfinite(f)
    pts = np.column_stack([lo[ok], glat[ok]]); val = f[ok]
    # pad periodically in lon so the interpolation wraps cleanly across +/-180
    pts = np.vstack([pts, pts + [360, 0], pts - [360, 0]]); val = np.concatenate([val, val, val])
    lin = griddata(pts, val, (LO, LA), method='linear')
    nn = griddata(pts, val, (LO, LA), method='nearest')
    out = np.where(np.isfinite(lin), lin, nn)
    # Decide what may be DRAWN from the source mask geometry, not from a distance: an output cell
    # is kept only if its nearest source cell is itself valid. A distance threshold cannot do this
    # job -- the masked coastal band is ~2*Delta wide and so widens with coarsening, and any
    # threshold that also scales with Delta refills it by interpolation. Measured with the old
    # 1.5*Delta rule: blanked area stayed at 40-42% across Delta=0.4-1.5 while the true masked area
    # grew 45->53%, i.e. at 1.5 deg a tenth of the map was painted over masked source and the
    # land/exclusion imprint visibly failed to scale with resolution.
    ap = np.column_stack([lo.ravel(), glat.ravel()]); av = ok.ravel().astype(float)
    fin = np.isfinite(ap[:, 0]) & np.isfinite(ap[:, 1]); ap, av = ap[fin], av[fin]
    ap = np.vstack([ap, ap + [360, 0], ap - [360, 0]]); av = np.concatenate([av, av, av])
    axy = np.column_stack([ap[:, 0] * np.cos(np.deg2rad(ap[:, 1])), ap[:, 1]])
    dn, idx = cKDTree(axy).query(QXY)
    # the distance cap remains only as a backstop for cells beyond the domain edge entirely
    keep = (av[idx] > 0.5) & (dn <= 1.5 * dx_deg)
    return np.where(keep.reshape(LO.shape), out, np.nan)

DX_MAP = SPACING[FACTORS.index(MAP_FA)]
GT_r = regrid_ll(GT, lon, lat, DX_MAP)
GP_r = regrid_ll(GP, lon, lat, DX_MAP)"""

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

# (a) global forcing maps STACKED: diagnosed (top) / ANN (bottom).
# Use a low colour limit so the faint open-ocean forcing shows colour rather than washing out to
# white -- |G| is intermittent (strong only in jets), so a high limit leaves most of the ocean
# near-white.
vmax = np.nanpercentile(np.abs(GT_r), 93); axs = []   # relaxed range (less saturated)
for r, (arr, ttl) in enumerate([(GT_r, 'diagnosed (CM2.6)'), (GP_r, 'ANN')]):
    ax = fig.add_subplot(gmap[r], projection=proj); axs.append(ax)
    im = ax.pcolormesh(loni, lati, arr, cmap=bal, vmin=-vmax, vmax=vmax, zorder=1,
                       transform=ccrs.PlateCarree(), rasterized=True)
    land(ax); ax.set_global(); ax.set_title(ttl, fontsize=11)
    ax.plot([GS_BOX[0], GS_BOX[1], GS_BOX[1], GS_BOX[0], GS_BOX[0]],
            [GS_BOX[2], GS_BOX[2], GS_BOX[3], GS_BOX[3], GS_BOX[2]], 'k-', lw=0.8, zorder=5,
            transform=ccrs.PlateCarree())
fig.colorbar(im, ax=axs, orientation='vertical', shrink=0.85, pad=0.02,
             label=r'$G=-\\mathbf{u}^*\\!\\cdot\\!\\nabla\\rho$ at %.0f m, $\\Delta=%.1f^\\circ$' % (zl[k], SPACING[FACTORS.index(MAP_FA)]))

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

# (c) grouped heatmaps along the BOTTOM (wide): per scale a box of 4 sub-bins along|across|force|ape
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
    ax.set_xticks(xc); ax.set_xticklabels(tlab, fontsize=8)              # al/ac/fc/pe along the bottom
    ax.set_xlabel(r'al = along-$\\nabla\\rho$,   ac = across-$\\nabla\\rho$,   fc = forcing $G$,   pe = APE release', fontsize=9)
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
for m in METRICS:
    print(f'depth-mean R2F_{m}_away:', {SPACING[i]: round(float(np.nanmean(R2[m][i])), 3) for i in range(4)})"""

R2MAP = """# (d) SECOND FIGURE -- where the skill lives. Per-point flux R^2 at one depth, one panel per
# coarse grid, so the decline with coarsening can be read regionally rather than as a single number.
# R2F_map is the stored 1 - <|f-F|^2>_t / <|F|^2>_t, reduced over the test snapshots at each point and
# uncentered to match the training loss. It is well behaved: only 2-7% of wet points are negative
# (they are weak-flux points where there is little to predict), so we clip the scale at 0 and print
# the fraction below it for the caption.
# Read the maps from skill-test rather than skill-test-forceape: the two are byte-identical at
# factors 9/12/15, but the forceape factor-4 was run on an 8-snapshot time subsample (NTIME4) to fit
# in memory, so its per-point R^2 is the noisier estimate. Everything else in this figure set uses
# forceape, which is the only place the force/APE metrics live.
MAPROOT = ROOT.replace('skill-test-forceape', 'skill-test')
skm = {fa: xr.open_dataset(f'{MAPROOT}/factor-{fa}.nc') for fa in FACTORS}
# Everything below the colour floor (0.2, not 0 -- see vmin below) is drawn blue, NOT grey: grey
# reads as land on these maps. So blue means "R^2 < 0.2", which includes the small negative tail.
reds = plt.get_cmap('Reds').copy(); reds.set_under('#2c7fb8'); reds.set_bad('white')
fig2 = plt.figure(figsize=(12.5, 6.0), constrained_layout=True)
gs2 = fig2.add_gridspec(2, 2)
axs2 = []
for i, fa in enumerate(FACTORS):
    pmi = xr.open_dataset(f'{PARAM}/factor-{fa}/param.nc'); gi = create_grid(pmi)
    zli = skm[fa]['zl'].values; ki = int(np.argmin(np.abs(zli - MAP_DEPTH)))
    wl = pmi.wet.isel(zl=ki) if 'zl' in pmi.wet.dims else pmi.wet
    w2 = np.asarray(xr.where(propagate_mask(wl, gi, niter=2) < 0.5, np.nan, 1.))
    r2 = np.where(w2 > 0.5, skm[fa]['R2F_map'].isel(zl=ki).values, np.nan)
    good = r2[np.isfinite(r2)]
    print(f'D={SPACING[i]}deg z={zli[ki]:.0f}m  median R2 ={np.median(good):.2f}  '
          f'frac<0.2 (blue) ={np.mean(good < 0.2):.3f}  frac<0 ={np.mean(good < 0):.3f}')
    ax = fig2.add_subplot(gs2[i // 2, i % 2], projection=proj); axs2.append(ax)
    im3 = ax.pcolormesh(loni, lati,
                        regrid_ll(r2, pmi['geolon'].values, pmi['geolat'].values, SPACING[i]),
                        cmap=reds, vmin=0.2, vmax=1., zorder=1,   # most of the ocean is >0.6; a
                        # 0-1 scale wastes the range and flattens the regional structure
                        transform=ccrs.PlateCarree(), rasterized=True)
    land(ax); ax.set_global()
    ax.set_title(r'$\\Delta=%.1f^\\circ$   (median $R^2$ = %.2f)' % (SPACING[i], np.nanmedian(r2)),
                 fontsize=11)
fig2.colorbar(im3, ax=axs2, orientation='vertical', shrink=0.7, pad=0.02, extend='min',
              label=r'per-point $R^2$ of $\\mathbf{F}^\\rho_h$ at %.0f m' % zl[k])
fig2.suptitle('Regional structure of the offline flux skill (held-out test split)', fontsize=13)
fig2.savefig('offline_skill_maps.pdf', bbox_inches='tight', dpi=150)
fig2.savefig('offline_skill_maps.png', bbox_inches='tight', dpi=150)
print('rendered offline_skill_maps')"""

CELLS = [
    ('md', "# Offline-skill figure (Section~3.3)\n\n"
           "Composite, in the style of Perezhogin et al. (2025) Fig. 1 but for the buoyancy "
           "problem, on the held-out test split:\n"
           "- global Robinson maps of the **forcing the scheme applies**, "
           "$G=-\\mathbf{u}^*\\cdot\\nabla\\rho$ (diagnosed vs ANN) at 500 m;\n"
           "- a **Gulf Stream** zoom of the along-/across-gradient flux (diagnosed vs ANN);\n"
           "- **grouped heatmaps**: each coarse-grid spacing is a box of four sub-bins "
           "(along | across | forcing | APE release) over depth, for $R^2$ and correlation.\n\n"
           "Note we do *not* show $\\nabla_h\\cdot\\mathbf{F}^\\rho_h$: the Ferrari division by "
           "$|\\nabla_3\\rho|$ precedes the derivative, so the horizontal flux divergence is not "
           "what the model feels (flux-decomposition appendix).\n\n"
           "A second figure, `offline_skill_maps.pdf`, then shows the **per-point** flux $R^2$ at "
           "525 m for each coarse grid, so the decline with coarsening can be read regionally.\n\n"
           "Run top-to-bottom with the Pavel_Container kernel; copy `offline_skill.pdf` and "
           "`offline_skill_maps.pdf` into the paper repo `figures/`."),
    ('code', SETUP), ('code', LOAD), ('code', FORCE), ('code', PLOT), ('code', R2MAP),
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

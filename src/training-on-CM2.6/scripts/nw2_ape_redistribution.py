"""Does the 1/4-degree dAPE really redistribute rather than vanish? Maps + zonal means.

nw2_ape_response.py found the AREA-MEAN dAPE collapsing toward zero from 1/2 to 1/4 degree while
the map rms stays large. This figure makes that point directly:
  row 1       dAPE maps at 1/4 deg (corrected functional, last 5 windows), each annotated with its
              area-mean and rms -- large-amplitude dipoles, small residual mean;
  row 2       dEKE maps at 1/4 deg (rho0-weighted depth-INTEGRATED resolved eddy KE) -- the
              complementary story: where each closure kills or spares the resolved eddies.
Both map rows use the FIXED colour scales of Perezhogin's Figure-2-and-S1.ipynb (NW2 energy
panels): dAPE PuOr_r SymLog(linthresh 1e5, +-1e7 J/m2); dEKE curl SymLog(linthresh 1e4,
+-1e6 J/m2) -- directly comparable with the published momentum-paper maps. Note his dEKE is
(reference - control), i.e. what the MISSING eddies carry; ours is (closure - control), so our
EKE row sitting mostly inside the linear zone is itself the message;
  bottom      (f) zonal-mean dAPE vs latitude, 1/4 deg solid vs 1/2 deg dashed, and (g) the
              south-to-north cumulative area integral of dAPE in Joules: a curve that wanders and
              returns to zero is pure rearrangement, an endpoint far from zero is net change. The
              endpoints cross-check against the ocean.stats time series (nw2_timeseries.py):
              ANN C=1 at 1/4 deg ends at ~+0.8e18 J, matching stats' dAPE(run)-dAPE(bare).

The sharp zonal jumps in the GM maps near 54.5S (and the on/off edge at the ~69S shelf break) are
REAL, not artifacts (checked 2026-08-10): the pinning term cancels exactly in run differences, and
the jump rows coincide with where GM has outcropped 1-2 more interfaces than the no-closure run
(interface 8, reference depth ~1050 m, sits AT the surface in GM1600 and at depth in bare). The
quadratic functional amplifies that displacement, and GM's eddy suppression keeps its time-mean
outcrop front razor-sharp -- strong GM physically relocates the ACC outcrop front."""
import numpy as np, xarray as xr, string
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, cmocean
from nw2_common import emean, load_grid, load_e_rest, ape_map, eke_map

RUNGS = [("1/2$^\\circ$", "/scratch/db194/mom6/jul2026_nw2"),
         ("1/4$^\\circ$", "/scratch/db194/mom6/jul2026_nw2_R4")]
ORDER = [("GM400", "GM $\\kappa$=400"), ("GM1600", "GM $\\kappa$=1600"), ("MEKE", "MEKE"),
         ("ANN_c1p0", "ANN $C$=1"), ("ANN_c1p5", "ANN $C$=1.5")]
COL = {"GM400": "#6baed6", "GM1600": "#08306b", "MEKE": "#1b9e77",
       "ANN_c1p0": "#e6550d", "ANN_c1p5": "#a63603"}

data = {}
for lab, base in RUNGS:
    rho_l, lat, lon = load_grid(base)
    drho = np.diff(rho_l)
    e_rest = load_e_rest(base)
    A0 = ape_map(emean(base, "bare"), e_rest, drho)
    data[lab] = (lat, lon, {r: ape_map(emean(base, r), e_rest, drho) - A0 for r, _ in ORDER})
    print(f"--- {lab} APE loaded", flush=True)

RHO0 = 1035.0
B4 = RUNGS[1][1]
E0 = eke_map(B4, "bare", integrated=True)
eres4 = {r: RHO0 * (eke_map(B4, r, integrated=True) - E0) for r, _ in ORDER}
print("--- 1/4 EKE loaded", flush=True)

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8.5})
n = len(ORDER)
fig = plt.figure(figsize=(2.45 * n, 13.2), constrained_layout=True)
gs = fig.add_gridspec(3, n, height_ratios=[1.9, 1.9, 1.0])
lat4, lon4, res4 = data["1/4$^\\circ$"]
w4 = np.cos(np.deg2rad(lat4))[:, None]

def map_panel(a, f, norm, cmap, first_col, last_maprow):
    im = a.pcolormesh(lon4, lat4, f, norm=norm, cmap=cmap, shading="auto", rasterized=True)
    a.set_aspect("equal")
    lats = [-60, -40, -20, 0, 20, 40, 60]
    a.set_yticks(lats)
    a.set_yticklabels([f"${abs(v)}^\\circ$S" if v < 0 else (f"${v}^\\circ$N" if v > 0 else "$0^\\circ$")
                       for v in lats] if first_col else [""] * 7)
    a.set_xticks([10, 30, 50])
    a.set_xticklabels([f"${x}^\\circ$E" for x in [10, 30, 50]] if last_maprow else [""] * 3)
    return im

def wstats(f):
    m = np.nansum(f * w4) / np.nansum(w4 * np.isfinite(f))
    return m, np.sqrt(np.nansum(f**2 * w4) / np.nansum(w4 * np.isfinite(f)))

# Perezhogin Figure-2-and-S1.ipynb scales, verbatim
norm_a = mpl.colors.SymLogNorm(linthresh=1e5, vmin=-1e7, vmax=1e7, base=10)
norm_e = mpl.colors.SymLogNorm(linthresh=1e4, vmin=-1e6, vmax=1e6, base=10)
axA, axE = [], []
for j, (r, rlab) in enumerate(ORDER):
    a = fig.add_subplot(gs[0, j]); axA.append(a)
    imA = map_panel(a, res4[r], norm_a, plt.cm.PuOr_r, j == 0, False)
    m, rms = wstats(res4[r])
    a.set_title(f"{rlab}\nmean {m/1e3:+.0f}, rms {rms/1e3:.0f} kJ m$^{{-2}}$", loc="left", fontsize=9.5)
    a.text(0.04, 0.975, f"({string.ascii_lowercase[j]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")
    a = fig.add_subplot(gs[1, j]); axE.append(a)
    imE = map_panel(a, eres4[r], norm_e, cmocean.cm.curl, j == 0, True)
    m, rms = wstats(eres4[r])
    a.set_title(f"mean {m/1e3:+.1f}, rms {rms/1e3:.1f} kJ m$^{{-2}}$", loc="left", fontsize=9.5)
    a.text(0.04, 0.975, f"({string.ascii_lowercase[n+j]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")
axA[0].set_ylabel("1/4$^\\circ$ latitude")
axE[0].set_ylabel("1/4$^\\circ$ latitude")
fig.colorbar(imA, ax=axA, orientation="horizontal", extend="both", aspect=60, pad=0.02,
             shrink=0.75).set_label("$\\Delta$APE vs no closure at 1/4$^\\circ$  [J m$^{-2}$]", fontsize=10)
fig.colorbar(imE, ax=axE, orientation="horizontal", extend="both", aspect=60, pad=0.02,
             shrink=0.75).set_label("$\\rho_0\\,\\Delta$ depth-int eddy KE vs no closure at 1/4$^\\circ$  [J m$^{-2}$]", fontsize=10)

RE = 6.371e6
def cumint(lat, lon, f):
    """South-to-north cumulative area integral of f [J/m2] -> [J]."""
    dphi = np.deg2rad(lat[1] - lat[0]); dlam = np.deg2rad(lon[1] - lon[0])
    dA = RE**2 * np.cos(np.deg2rad(lat))[:, None] * dphi * dlam * np.isfinite(f)
    return np.cumsum(np.nansum(f * dA, axis=1))

gsb = gs[2, :].subgridspec(1, 2)
az = fig.add_subplot(gsb[0]); ac = fig.add_subplot(gsb[1])
from matplotlib.lines import Line2D
for r, rlab in ORDER:
    for (lab, _), ls in zip(RUNGS, ["--", "-"]):
        lat, lon, res = data[lab]
        az.plot(lat, np.nanmean(res[r], axis=1) / 1e3, ls, color=COL[r], lw=1.5)
        ac.plot(lat, cumint(lat, lon, res[r]) / 1e18, ls, color=COL[r], lw=1.5)
for a, yl, ttl in [(az, "zonal-mean $\\Delta$APE  [kJ m$^{-2}$]", "zonal mean"),
                   (ac, "$\\int\\Delta$APE $dA$ from south  [$10^{18}$ J]", "cumulative area integral")]:
    a.axhline(0, color="k", lw=0.6, alpha=0.4)
    a.set_xlabel("latitude"); a.set_ylabel(yl)
    a.set_xlim(lat4[0], lat4[-1])
    a.spines[["top", "right"]].set_visible(False)
    a.set_title(f"{ttl}: 1/2$^\\circ$ dashed, 1/4$^\\circ$ solid", loc="left")
for k, a in enumerate([az, ac]):
    a.text(0.01, 0.97, f"({string.ascii_lowercase[2*n+k]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")
handles = [Line2D([], [], color="k", ls="--", lw=1.5), Line2D([], [], color="k", ls="-", lw=1.5)] + \
          [Line2D([], [], color=COL[r], lw=2.2) for r, _ in ORDER]
labels = ["1/2$^\\circ$", "1/4$^\\circ$"] + [rlab for _, rlab in ORDER]
az.legend(handles, labels, frameon=False, ncol=4, fontsize=8.5, loc="lower right")

png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_ape_redistribution.png"
fig.savefig(png, dpi=140, bbox_inches="tight")
print("wrote", png)

print(f"\n{'run':<10}" + "".join(f"{l+' mean':>13}{l+' rms':>12}" for l, _ in RUNGS) + "   [kJ/m2]")
for r, _ in ORDER:
    row = ""
    for lab, _ in RUNGS:
        lat, lon, res = data[lab]
        w = np.cos(np.deg2rad(lat))[:, None]
        m = np.nansum(res[r] * w) / np.nansum(w * np.isfinite(res[r]))
        rms = np.sqrt(np.nansum(res[r] ** 2 * w) / np.nansum(w * np.isfinite(res[r])))
        row += f"{m/1e3:>13.1f}{rms/1e3:>12.1f}"
    print(f"{r:<10}{row}")

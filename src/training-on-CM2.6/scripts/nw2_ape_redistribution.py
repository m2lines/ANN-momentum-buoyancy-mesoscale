"""Does the 1/4-degree dAPE really redistribute rather than vanish? Maps + zonal means.

nw2_ape_response.py found the AREA-MEAN dAPE collapsing toward zero from 1/2 to 1/4 degree while
the map rms stays large. This figure makes that point directly:
  top row     dAPE maps at 1/4 deg (corrected functional, last 5 windows), each annotated with its
              area-mean and rms -- large-amplitude dipoles, small residual mean;
  bottom      (f) zonal-mean dAPE vs latitude, 1/4 deg solid vs 1/2 deg dashed, and (g) the
              south-to-north cumulative area integral of dAPE in Joules: a curve that wanders and
              returns to zero is pure rearrangement, an endpoint far from zero is net change. The
              endpoints cross-check against the ocean.stats time series (nw2_timeseries.py):
              ANN C=1 at 1/4 deg ends at ~+0.8e18 J, matching stats' dAPE(run)-dAPE(bare)."""
import numpy as np, xarray as xr, string
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from nw2_common import emean, load_grid, load_e_rest, ape_map

RUNGS = [("1/2$^\\circ$", "/scratch/db194/mom6/jul2026_nw2"),
         ("1/4$^\\circ$", "/scratch/db194/mom6/jul2026_nw2_R4")]
ORDER = [("GM400", "GM $\\kappa$=400"), ("GM1600", "GM $\\kappa$=1600"), ("MEKE", "MEKE"),
         ("ANN_c1p0", "ANN $C$=1"), ("ANN_c1p5", "ANN $C$=1.5")]
COL = {"GM400": "#6baed6", "GM1600": "#08306b", "MEKE": "#9ecae1",
       "ANN_c1p0": "#e6550d", "ANN_c1p5": "#a63603"}

data = {}
for lab, base in RUNGS:
    rho_l, lat, lon = load_grid(base)
    drho = np.diff(rho_l)
    e_rest = load_e_rest(base)
    A0 = ape_map(emean(base, "bare"), e_rest, drho)
    data[lab] = (lat, lon, {r: ape_map(emean(base, r), e_rest, drho) - A0 for r, _ in ORDER})
    print(f"--- {lab} loaded", flush=True)

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 8.5})
n = len(ORDER)
fig = plt.figure(figsize=(2.45 * n, 8.6), constrained_layout=True)
gs = fig.add_gridspec(2, n, height_ratios=[1.9, 1.0])
lat4, lon4, res4 = data["1/4$^\\circ$"]
w4 = np.cos(np.deg2rad(lat4))[:, None]
allv = np.abs(np.concatenate([res4[r].ravel() for r, _ in ORDER])) / 1e6
allv = allv[np.isfinite(allv)]
norm = mpl.colors.SymLogNorm(linthresh=np.percentile(allv, 60),
                             vmin=-np.percentile(allv, 99.8), vmax=np.percentile(allv, 99.8), base=10)
for j, (r, rlab) in enumerate(ORDER):
    a = fig.add_subplot(gs[0, j])
    im = a.pcolormesh(lon4, lat4, res4[r] / 1e6, norm=norm, cmap=plt.cm.PuOr_r,
                      shading="auto", rasterized=True)
    a.set_aspect("equal")
    mean = np.nansum(res4[r] * w4) / np.nansum(w4 * np.isfinite(res4[r]))
    rms = np.sqrt(np.nansum(res4[r] ** 2 * w4) / np.nansum(w4 * np.isfinite(res4[r])))
    a.set_title(f"{rlab}\nmean {mean/1e3:+.0f}, rms {rms/1e3:.0f} kJ m$^{{-2}}$",
                loc="left", fontsize=9.5)
    lats = [-60, -40, -20, 0, 20, 40, 60]
    a.set_yticks(lats)
    a.set_yticklabels([f"${abs(v)}^\\circ$S" if v < 0 else (f"${v}^\\circ$N" if v > 0 else "$0^\\circ$")
                       for v in lats] if j == 0 else [""] * 7)
    a.set_xticks([10, 30, 50]); a.set_xticklabels([f"${x}^\\circ$E" for x in [10, 30, 50]])
    a.text(0.04, 0.975, f"({string.ascii_lowercase[j]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")
    if j == 0:
        a.set_ylabel("1/4$^\\circ$ latitude")
fig.colorbar(im, ax=[fig.axes[j] for j in range(n)], orientation="horizontal", extend="both",
             aspect=60, pad=0.02, shrink=0.75).set_label(
    "$\\Delta$APE vs no closure at 1/4$^\\circ$  [MJ m$^{-2}$]", fontsize=10)

RE = 6.371e6
def cumint(lat, lon, f):
    """South-to-north cumulative area integral of f [J/m2] -> [J]."""
    dphi = np.deg2rad(lat[1] - lat[0]); dlam = np.deg2rad(lon[1] - lon[0])
    dA = RE**2 * np.cos(np.deg2rad(lat))[:, None] * dphi * dlam * np.isfinite(f)
    return np.cumsum(np.nansum(f * dA, axis=1))

gsb = gs[1, :].subgridspec(1, 2)
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
    a.text(0.01, 0.97, f"({string.ascii_lowercase[n+k]})", transform=a.transAxes, va="top",
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

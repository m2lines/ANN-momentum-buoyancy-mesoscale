"""NW2 domain-integrated APE and KE time series, both rungs, all canonical closures.

Source is each run's own ocean.stats.nc (5-day records, MOM6's global energy accounting: APE is
referenced to the adiabatically-flattened -H0 state, the same reference nw2_common.ape_map uses).
All runs at a rung branch from the same day-30001 restart, so plotting ANOMALIES from the branch
point shows the response to switching the closure on directly; the no-closure run (gray) carries
the shared drift/seasonality against which the closures read."""
import numpy as np, xarray as xr
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

RUNGS = [("1/2$^\\circ$", "/scratch/db194/mom6/jul2026_nw2"),
         ("1/4$^\\circ$", "/scratch/db194/mom6/jul2026_nw2_R4")]
RUNS = ["bare", "GM400", "GM1600", "MEKE", "ANN_c1p0", "ANN_c1p5"]
COL = {"bare": "#8c8c8c", "GM400": "#6baed6", "GM1600": "#08306b", "MEKE": "#1b9e77",
       "ANN_c1p0": "#e6550d", "ANN_c1p5": "#a63603"}
LAB = {"bare": "no closure", "GM400": "GM $\\kappa$=400", "GM1600": "GM $\\kappa$=1600",
       "MEKE": "MEKE", "ANN_c1p0": "ANN $C$=1", "ANN_c1p5": "ANN $C$=1.5"}

def series(base, r):
    d = xr.open_dataset(f"{base}/{r}/output/ocean.stats.nc", decode_times=False)
    t = (d["Time"].values - d["Time"].values[0]) / 365.0
    ape = d["APE"].sum("Interface").values
    ke = d["KE"].sum("Layer").values
    d.close()
    return t, ape - ape[0], ke - ke[0]

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "legend.fontsize": 8.5})
fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, constrained_layout=True)
for j, (lab, base) in enumerate(RUNGS):
    for r in RUNS:
        t, dape, dke = series(base, r)
        kw = dict(color=COL[r], lw=1.1 if r != "bare" else 1.4,
                  ls="-" if r != "bare" else "--", label=LAB[r])
        ax[0, j].plot(t, dape / 1e18, **kw)
        ax[1, j].plot(t, dke / 1e17, **kw)
    ax[0, j].set_title(f"{lab}", loc="left")
for a in ax.ravel():
    a.axhline(0, color="k", lw=0.6, alpha=0.4)
    a.spines[["top", "right"]].set_visible(False)
ax[0, 0].set_ylabel("$\\Delta$APE since branch  [$10^{18}$ J]")
ax[1, 0].set_ylabel("$\\Delta$KE since branch  [$10^{17}$ J]")
for a in ax[1]:
    a.set_xlabel("years since branch (day 30001)")
ax[0, 0].legend(frameon=False, ncol=2)
for k, a in enumerate(ax.ravel()):
    a.text(0.02, 0.97, f"({'abcd'[k]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")

png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_timeseries.png"
fig.savefig(png, dpi=150, bbox_inches="tight")
print("wrote", png)

# final-2-year means, for the record
print(f"\n{'run':<10}" + "".join(f"{l+' dAPE':>14}{l+' dKE':>13}" for l, _ in RUNGS) + "   [1e18 J / 1e17 J]")
for r in RUNS:
    row = ""
    for _, base in RUNGS:
        t, dape, dke = series(base, r)
        m = t > t[-1] - 2
        row += f"{dape[m].mean()/1e18:>14.2f}{dke[m].mean()/1e17:>13.2f}"
    print(f"{r:<10}{row}")

"""APE and KE against time for the NW2 Phase-1 set, straight from ocean.stats.nc.

This is the cheapest and in some ways the best diagnostic we have: MOM6 writes APE (per interface)
and KE (per layer) every ENERGYSAVEDAYS=5 days, so 1441 samples per run, and it computes APE
against its OWN reference -- H0, the interface heights the state would relax to if adiabatically
flattened to minimum potential energy. That is a physically defined reference, unlike the
horizontal-mean convention the map estimator has to adopt, and it is identical across runs.

Panels: (a) total APE(t), (b) total KE(t), (c) APE anomaly relative to the unparameterized run,
which is the actual response signal and shows how quickly it emerges."""
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/jul2026_nw2"
RUNS = [("bare", "no closure", "#8c8c8c"), ("MEKE", "MEKE", "#9ecae1"),
        ("GM400", "GM $\\kappa$=400", "#4292c6"), ("GM1600", "GM $\\kappa$=1600", "#08306b"),
        ("ANN_c1p0", "ANN $C$=1", "#e6550d"), ("ANN_c2p0", "ANN $C$=2", "#a63603")]

d = {}
for r, lab, c in RUNS:
    ds = xr.open_dataset(f"{B}/{r}/output/ocean.stats.nc", decode_times=False)
    t = ds["Time"].values
    d[r] = dict(t=t, ape=ds["APE"].values.sum(axis=1), ke=ds["KE"].values.sum(axis=1), lab=lab, c=c)
    ds.close()

t0 = d["bare"]["t"][0]
print(f"{'run':<10}{'APE final [J]':>15}{'dAPE vs bare':>15}{'KE final [J]':>15}{'dKE %bare':>11}")
for r, lab, c in RUNS:
    s = d[r]
    n = min(len(s["ape"]), len(d["bare"]["ape"]))
    ape_f = s["ape"][-20:].mean(); ke_f = s["ke"][-20:].mean()
    b_ape = d["bare"]["ape"][-20:].mean(); b_ke = d["bare"]["ke"][-20:].mean()
    print(f"{r:<10}{ape_f:>15.4e}{ape_f-b_ape:>+15.3e}{ke_f:>15.4e}{100*ke_f/b_ke:>10.1f}%")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
for r, lab, c in RUNS:
    s = d[r]; yr = (s["t"] - t0) / 360.0
    ax[0].plot(yr, s["ape"] / 1e18, color=c, lw=1.4, label=lab)
    ax[1].plot(yr, s["ke"] / 1e15, color=c, lw=1.4, label=lab)
    if r != "bare":
        n = min(len(s["ape"]), len(d["bare"]["ape"]))
        ax[2].plot(yr[:n], (s["ape"][:n] - d["bare"]["ape"][:n]) / 1e18, color=c, lw=1.4, label=lab)
ax[2].axhline(0, color="#8c8c8c", lw=1.0)
for a, ttl, yl in [(ax[0], "(a) total APE", "APE [$10^{18}$ J]"),
                   (ax[1], "(b) total KE", "KE [$10^{15}$ J]"),
                   (ax[2], "(c) APE response (run $-$ no closure)", "$\\Delta$APE [$10^{18}$ J]")]:
    a.set_title(ttl, loc="left"); a.set_xlabel("years since branch"); a.set_ylabel(yl)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, ncol=2, fontsize=7.5)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_energy_timeseries.png"
fig.savefig(png, dpi=150); print("\nwrote", png)

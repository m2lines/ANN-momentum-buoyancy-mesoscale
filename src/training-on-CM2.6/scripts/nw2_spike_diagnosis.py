"""Are the KE spikes in the higher-C ANN runs physical, or numerical trouble?

ocean.stats.nc carries exactly what is needed to tell: `Ntrunc` (velocity truncations applied that
step -- nonzero means the model had to clip a velocity, i.e. numerical distress) and the two CFL
diagnostics, alongside KE resolved BY LAYER, which localises where a spike lives in the vertical.

A physical eddy-energy excursion should show no truncations, CFL well under 1, and KE spread over
the layers that carry the flow. A numerical one shows truncations coincident with the spikes and
KE concentrated in one or two layers -- most often the thinnest."""
import numpy as np, xarray as xr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/jul2026_nw2"
RUNS = ["bare", "GM400", "GM1600", "MEKE", "ANN_c1p0", "ANN_c1p5", "ANN_c2p0"]

d = {}
for r in RUNS:
    ds = xr.open_dataset(f"{B}/{r}/output/ocean.stats.nc", decode_times=False)
    d[r] = dict(t=(ds["Time"].values - ds["Time"].values[0]) / 360.0,
                ntr=ds["Ntrunc"].values, ke=ds["KE"].values,
                cfl=ds["max_CFL_trans"].values, cfl_lin=ds["max_CFL_lin"].values)
    ds.close()

print(f"{'run':<10}{'tot Ntrunc':>12}{'steps>0':>9}{'max CFL_tr':>12}{'max CFL_lin':>13}"
      f"{'KE p99/median':>15}{'layer of max KE':>17}")
for r in RUNS:
    s = d[r]; ke = s["ke"].sum(axis=1)
    kl = s["ke"][np.argmax(ke)]                       # layer profile at the peak-KE time
    print(f"{r:<10}{int(s['ntr'].sum()):>12}{int((s['ntr']>0).sum()):>9}{s['cfl'].max():>12.3f}"
          f"{s['cfl_lin'].max():>13.3f}{np.percentile(ke,99)/np.median(ke):>15.2f}"
          f"{int(np.argmax(kl))+1:>10} of {s['ke'].shape[1]}")

# where does the excess KE live, at the spikes, for the ANN runs?
print("\nlayer share of KE at the 5 largest spikes (ANN runs), vs the run's own median profile:")
for r in ("ANN_c1p0", "ANN_c1p5", "ANN_c2p0"):
    s = d[r]; ke = s["ke"].sum(axis=1)
    top = np.argsort(ke)[-5:]
    prof_hi = s["ke"][top].mean(axis=0); prof_hi /= prof_hi.sum()
    prof_md = np.median(s["ke"], axis=0); prof_md /= prof_md.sum()
    k = int(np.argmax(prof_hi - prof_md))
    print(f"  {r:<10} spike KE concentrates in layer {k+1:2d} "
          f"({100*prof_hi[k]:.0f}% of KE at spikes vs {100*prof_md[k]:.0f}% typically)")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.4), constrained_layout=True)
COL = {"bare": "#8c8c8c", "GM400": "#4292c6", "GM1600": "#08306b", "MEKE": "#1b9e77",
       "ANN_c1p0": "#fd8d3c", "ANN_c1p5": "#e6550d", "ANN_c2p0": "#a63603"}
for r in RUNS:
    s = d[r]
    ax[0].plot(s["t"], s["ntr"], color=COL[r], lw=1.0, label=r)
    ax[1].plot(s["t"], s["cfl"], color=COL[r], lw=0.9)
ax[0].set_title("(a) velocity truncations per report", loc="left"); ax[0].set_ylabel("Ntrunc")
ax[1].set_title("(b) max CFL (transport)", loc="left"); ax[1].set_ylabel("CFL")
ax[1].axhline(1.0, color="k", ls=":", lw=1)
s = d["ANN_c2p0"]
for k in range(s["ke"].shape[1]):
    ax[2].plot(s["t"], s["ke"][:, k] / 1e15, lw=0.8, color=plt.cm.viridis(k / 14))
ax[2].set_title("(c) ANN $C$=2: KE by layer (dark=surface)", loc="left")
ax[2].set_ylabel("KE [$10^{15}$ J]")
for a in ax:
    a.set_xlabel("years since branch"); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, ncol=2, fontsize=7)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_spike_diagnosis.png"
fig.savefig(png, dpi=150); print("\nwrote", png)

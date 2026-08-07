"""NW2 closure comparison across the resolution ladder: 1/2 degree vs 1/4 degree.

The channel (Sec 4.1) found a two-regime result: at coarse resolution the closures separate on the
MEAN STATE, and at eddy-permitting resolution they separate on whether they DESTROY THE RESOLVED
EDDIES. This asks whether the same holds in NW2, where the two rungs are R2 (1/2 deg, marginally
eddy-permitting) and R4 (1/4 deg, clearly eddy-permitting), both branched from the same day-30001
state in the same configuration.

Metrics, all relative to each rung's own unparameterized run:
  EKE %bare  -- resolved eddy energy retained (snapshot variance, thickness weighted)
  rms d(e)   -- rms interface displacement, i.e. how much the closure moves the mean state
  ACC        -- circumpolar transport"""
import numpy as np, xarray as xr, glob
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["bare", "GM400", "GM1600", "MEKE", "ANN_c1p0", "ANN_c1p5"]
RUNGS = [("1/2$^\\circ$", "/scratch/db194/mom6/jul2026_nw2"),
         ("1/4$^\\circ$", "/scratch/db194/mom6/jul2026_nw2_R4")]
RHO0, NWIN = 1035.0, 5

def metrics(base, r):
    lm = sorted(glob.glob(f"{base}/{r}/output/longmean_*.nc"))[-NWIN:]
    sn = sorted(glob.glob(f"{base}/{r}/output/snapshots_*.nc"))[-NWIN:]
    e = np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values for f in lm], axis=0)
    acc = np.mean([float(np.nanmax(np.abs(np.nansum(
        xr.open_dataset(f, decode_times=False)["uh"].mean("time").values, axis=(0, 1))))/1e6) for f in lm])
    eke = []
    for f in sn:
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values; d.close()
        uc = 0.5*(u[..., :-1]+u[..., 1:]); vc = 0.5*(v[:, :, :-1, :]+v[:, :, 1:, :])
        KE = 0.5*np.nanmean((uc**2+vc**2)*h, 0)
        um, vm, hm = np.nanmean(uc, 0), np.nanmean(vc, 0), np.nanmean(h, 0)
        eke.append(float(np.nansum(KE - 0.5*(um**2+vm**2)*hm)/np.nansum(hm)))
    return e, np.mean(eke), acc

out = {}
for lab, base in RUNGS:
    e0, k0, a0 = metrics(base, "bare")
    for r in RUNS:
        e, k, a = metrics(base, r)
        out[(lab, r)] = dict(eke=100*k/k0, de=float(np.sqrt(np.nanmean((e-e0)**2))), acc=a)
    print(f"--- {lab}: bare EKE {k0:.4e}, ACC {a0:.1f} Sv", flush=True)

print(f"\n{'run':<10}" + "".join(f"{l+' EKE%':>12}{l+' rms de':>13}" for l, _ in RUNGS))
for r in RUNS:
    row = "".join(f"{out[(l,r)]['eke']:>11.0f}%{out[(l,r)]['de']:>13.1f}" for l, _ in RUNGS)
    print(f"{r:<10}{row}")

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "legend.fontsize": 8.5})
COL = {"GM400": "#6baed6", "GM1600": "#08306b", "MEKE": "#9ecae1",
       "ANN_c1p0": "#e6550d", "ANN_c1p5": "#a63603"}
LAB = {"GM400": "GM $\\kappa$=400", "GM1600": "GM $\\kappa$=1600", "MEKE": "MEKE",
       "ANN_c1p0": "ANN $C$=1", "ANN_c1p5": "ANN $C$=1.5"}
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
x = [0, 1]
for r in RUNS[1:]:
    ax[0].plot(x, [out[(l, r)]["eke"] for l, _ in RUNGS], "-o", color=COL[r], lw=1.8, ms=7, label=LAB[r])
    ax[1].plot(x, [out[(l, r)]["de"] for l, _ in RUNGS], "-o", color=COL[r], lw=1.8, ms=7)
ax[0].axhline(100, color="#8c8c8c", ls="--", lw=1.2, label="no closure")
for a, t, yl in [(ax[0], "(a) resolved eddy energy retained", "EKE  [% of no closure]"),
                 (ax[1], "(b) mean-state change", "rms interface displacement  [m]")]:
    a.set_xticks(x); a.set_xticklabels([l for l, _ in RUNGS]); a.set_title(t, loc="left")
    a.set_ylabel(yl); a.set_xlabel("grid spacing"); a.spines[["top", "right"]].set_visible(False)
    a.set_xlim(-0.25, 1.25)
ax[0].legend(frameon=False, ncol=2)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_ladder.png"
fig.savefig(png, dpi=150)
fig.savefig("/home/db194/mesoscale_b_ml_parameterization/figures/nw2_ladder.pdf", dpi=300, bbox_inches="tight")
print("wrote", png, "+ paper PDF")

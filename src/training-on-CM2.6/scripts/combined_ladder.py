"""Part 2 sec 4 CENTRAL FIGURE: baseline ladder vs 1/16 truth across the resolution ladder (1deg, 1/2deg, 1/4deg).
Two-regime thesis: at coarse res the mean state discriminates (no-param over-overturns; ANN fixes it untuned;
MEKE trades ACC); at eddy-permitting res the eddy preservation discriminates (MEKE kills resolved EKE; ANN keeps it).
Metrics: interior overturning Psi_max, outcrop band |Psi_min|, resolved EKE (snapshot variance, self-consistent)."""
import xarray as xr, numpy as np, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO0 = 1035.0
B = "/scratch/db194/mom6/feb2026"
def cd(res): return f"{B}/channel_extra_sponge_slow_woc_{res}"
RES = [("1p0", 1.0), ("p5", 0.5), ("p25", 0.25)]              # (dir tag, nominal dx deg)
CLOSURES = {                                                  # label -> (subdir pattern, spinup cutoff)
    "no-param":     ("tau_0.2_cb_0.0_cu_0.0", 4000),
    "ANN cb=1":     ("tau_0.2_cb_1.0_cu_0.0_neutral", 11500),  # canonical EXP_neutral_all4 re-runs
    "MEKE khf=0.4": ("tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500),
    "MEKE khf=0.8": ("tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.8", 11500),
}
TRUTH = (f"{cd('p0625')}/tau_0.2_cb_0.0_cu_0.0", 10000, 0.0625)

def dl(da, *k):
    for d in da.dims:
        if any(x == d or d.startswith(x) for x in k): return d
    return None
def twins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]
def acc(f):
    uh = xr.open_dataset(f, decode_times=False)["uh"]; return float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / RHO0 / 1e6).mean())
def psi(f):
    vmo = xr.open_dataset(f, decode_times=False)["vmo"]; vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
    P = (vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=dl(vmo, "rho2", "rho", "zl")); return float(P.max()), float(P.min())
def eke(rd, cut):
    v = []
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$", g) and int(g.split("_")[-1].split(".")[0]) >= cut):
        ds = xr.open_dataset(f, decode_times=False); u, w = ds["u"], ds["v"]; t = dl(u, "Time")
        v.append(0.5 * (float(u.var(dim=t).mean()) + float(w.var(dim=t).mean())))
    return np.mean(v) if v else np.nan

def metrics(rd, cut):
    wins = [w for w in twins(rd, cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    if not wins: return None
    a = [acc(f"{rd}/output/prog_tmean_{w}.nc") for w in wins]
    ps = [psi(f"{rd}/output/prog_rho_tmean_{w}.nc") for w in wins]
    return dict(acc=np.mean(a), pmax=np.mean([p[0] for p in ps]), pmin=np.mean([p[1] for p in ps]), eke=eke(rd, cut))

tr = metrics(TRUTH[0], TRUTH[1])
print(f"truth 1/16: Psi_max {tr['pmax']:.2f}  Psi_min {tr['pmin']:.2f}  ACC {tr['acc']:.3f}  EKE {tr['eke']:.3e}")
data = {c: {} for c in CLOSURES}
for c, (sub, cut) in CLOSURES.items():
    for tag, dx in RES:
        m = metrics(f"{cd(tag)}/{sub}", cut)
        if m: data[c][dx] = m
        print(f"  {c:<14} {tag:>4} ({dx}):  " + ("--" if not m else f"Psi_max {m['pmax']:.2f}  Psi_min {m['pmin']:.2f}  ACC {m['acc']:.3f}  EKE {m['eke']:.3e}"))

# --- figure: 3 panels, metric vs resolution ---
xs = [dx for _, dx in RES]
sty = {"no-param": ("#8c8c8c", "o"), "ANN cb=1": ("#e6550d", "s"), "MEKE khf=0.4": ("#4292c6", "^"), "MEKE khf=0.8": ("#08519c", "v")}
fig, axg = plt.subplots(2, 2, figsize=(11, 8)); ax = axg.ravel()
def line(a, key, sign=1):
    for c, (col, mk) in sty.items():
        xy = [(dx, sign * data[c][dx][key]) for dx in xs if dx in data[c]]
        if xy: a.plot(*zip(*xy), col, marker=mk, label=c, lw=1.8, ms=6)
    a.invert_xaxis(); a.set_xlabel("resolution (° grid spacing)"); a.set_xticks(xs); a.set_xticklabels(["1°", "½°", "¼°"])
line(ax[0], "pmax"); ax[0].axhline(tr["pmax"], color="k", ls="--", lw=1, label="1/16° truth")
ax[0].set_ylabel("Interior overturning $\\Psi$ [Sv]"); ax[0].set_title("Mean-state overturning")
line(ax[1], "eke"); ax[1].axhline(tr["eke"], color="k", ls="--", lw=1)
ax[1].set_ylabel("Resolved EKE [m²/s²]"); ax[1].set_title("Resolved eddy energy (the clean ANN win)"); ax[1].set_yscale("log")
line(ax[2], "acc"); ax[2].axhline(tr["acc"], color="k", ls="--", lw=1)
ax[2].set_ylabel("ACC transport [Sv]"); ax[2].set_title("ACC transport")
line(ax[3], "pmin", -1); ax[3].axhline(-tr["pmin"], color="k", ls="--", lw=1)
ax[3].set_ylabel("Outcrop band $|\\Psi|$ [Sv]"); ax[3].set_title("Eddy-driven overturning band")
ax[0].legend(fontsize=8, loc="best")
fig.suptitle("Channel closures vs 1/16° truth across resolution (τ=0.2): the ANN preserves resolved eddies at every resolution (top-right); MEKE kills them. Mean state: ANN good at coarse, over-corrects at fine.", fontsize=9)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/combined_ladder.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print(f"\nwrote {png}")

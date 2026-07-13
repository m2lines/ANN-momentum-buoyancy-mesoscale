"""ANN cb-sweep across resolution (Part 2 sec 4): does the ANN's dose-response explain the fine-res
over-correction, and does it preserve eddies at ALL cb or only low cb? cb=0 is no-param; cb>0 adds ANN flux.
Metrics vs cb, per resolution: interior overturning Psi_max, resolved EKE, ACC. Truth = 1/16 reference."""
import xarray as xr, numpy as np, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO0 = 1035.0
B = "/scratch/db194/mom6/feb2026"
def cd(res): return f"{B}/channel_extra_sponge_slow_woc_{res}"
RES = [("1p0", "1°", "#1b9e77"), ("p5", "½°", "#7570b3"), ("p25", "¼°", "#d95f02")]
CB = [0.0, 1.0, 2.0, 3.0, 4.0]
TRUTH = (f"{cd('p0625')}/tau_0.2_cb_0.0_cu_0.0", 10000)

def dl(da, *k):
    for d in da.dims:
        if any(x == d or d.startswith(x) for x in k): return d
    return None
def twins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]
def acc(f):
    uh = xr.open_dataset(f, decode_times=False)["uh"]; return float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / RHO0 / 1e6).mean())
def psimax(f):
    vmo = xr.open_dataset(f, decode_times=False)["vmo"]; vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
    return float((vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=dl(vmo, "rho2", "rho", "zl")).max())
def eke(rd, cut):
    v = []
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$", g) and int(g.split("_")[-1].split(".")[0]) >= cut):
        ds = xr.open_dataset(f, decode_times=False); u, w = ds["u"], ds["v"]; t = dl(u, "Time")
        v.append(0.5 * (float(u.var(dim=t).mean()) + float(w.var(dim=t).mean())))
    return np.mean(v) if v else np.nan
def metrics(rd, cut):
    wins = [w for w in twins(rd, cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    if not wins: return None
    return dict(acc=np.mean([acc(f"{rd}/output/prog_tmean_{w}.nc") for w in wins]),
                psi=np.mean([psimax(f"{rd}/output/prog_rho_tmean_{w}.nc") for w in wins]), eke=eke(rd, cut))

tr = metrics(*TRUTH)
print(f"truth 1/16: Psi_max {tr['psi']:.2f}  ACC {tr['acc']:.3f}  EKE {tr['eke']:.3e}\n")
D = {}
for tag, lab, _ in RES:
    D[tag] = {}
    for cb in CB:
        # cb=0 is the (ANN-free) no-param run; cb>0 uses the canonical EXP_neutral_all4 re-runs
        rd = f"{cd(tag)}/tau_0.2_cb_{cb}_cu_0.0" if cb == 0.0 else f"{cd(tag)}/tau_0.2_cb_{cb}_cu_0.0_neutral"
        m = metrics(rd, 4000 if cb == 0.0 else 11500)
        if m: D[tag][cb] = m
    row = "  ".join(f"cb{cb:.0f}:Psi{D[tag][cb]['psi']:.1f}/EKE{D[tag][cb]['eke']:.1e}" for cb in CB if cb in D[tag])
    print(f"{lab:>4}:  {row}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for tag, lab, col in RES:
    cbs = [c for c in CB if c in D[tag]]
    ax[0].plot(cbs, [D[tag][c]["psi"] for c in cbs], col, marker="o", label=lab, lw=1.8)
    ax[1].plot(cbs, [D[tag][c]["eke"] for c in cbs], col, marker="o", label=lab, lw=1.8)
    ax[2].plot(cbs, [D[tag][c]["acc"] for c in cbs], col, marker="o", label=lab, lw=1.8)
ax[0].axhline(tr["psi"], color="k", ls="--", lw=1, label="1/16° truth"); ax[0].set_ylabel("Interior $\\Psi$ [Sv]"); ax[0].set_title("Overturning vs cb")
ax[1].set_ylabel("Resolved EKE [m²/s²]"); ax[1].set_yscale("log"); ax[1].set_title("Eddy energy vs cb")
ax[2].axhline(tr["acc"], color="k", ls="--", lw=1); ax[2].set_ylabel("ACC [Sv]"); ax[2].set_title("ACC vs cb")
for a in ax: a.set_xlabel("ANN coefficient  cb  (0 = no-param)"); a.set_xticks(CB)
ax[0].legend(fontsize=8)
fig.suptitle("ANN cb-sweep (τ=0.2): no single fixed cb matches the mean state at all resolutions (ANN loses mean-state leverage as res increases), "
             "but resolved eddies survive at EVERY cb — unlike GM. cb=1 = reasonable-but-imperfect compromise.", fontsize=8)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/cb_sweep.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print(f"\nwrote {png}")

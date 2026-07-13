"""1deg channel baseline ladder vs the 1/16 truth (Part 2 sec 4, first slice).
Closures at 1deg (all tau=0.2): no-param / MEKE (KHTH_FAC sweep) / ANN, vs the eddy-resolving p0625 truth.
Metric per settled 2000-day window: ACC transport, residual overturning Psi(y,rho), domain KE (En).
Goal: pick the MEKE KHTH_FAC whose interior overturning matches truth (MEKE 'tuned', parallel to GM kappa-retune)."""
import xarray as xr, numpy as np, glob

RHO0 = 1035.0
BASE = "/scratch/db194/mom6/feb2026"
C1 = f"{BASE}/channel_extra_sponge_slow_woc_1p0"
# label -> (rundir, spin-up cutoff day: windows starting before this are dropped)
RUNS = {
    "truth 1/16 (p0625)": (f"{BASE}/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0", 8000),
    "no-param (1deg)":     (f"{C1}/tau_0.2_cb_0.0_cu_0.0",          4000),
    "ANN cb=1 (1deg)":     (f"{C1}/tau_0.2_cb_1.0_cu_0.0",          4000),
    "MEKE khf=0.2":        (f"{C1}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.2", 11500),
    "MEKE khf=0.4":        (f"{C1}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500),
    "MEKE khf=0.8":        (f"{C1}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.8", 11500),
}

def dim_like(da, *keys):
    for d in da.dims:
        if any(k == d or d.startswith(k) for k in keys): return d
    return None

def windows(rundir, cutoff):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rundir}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cutoff]

def acc_window(f):
    ds = xr.open_dataset(f, decode_times=False); uh = ds["uh"]
    return float((uh.sum(dim=[dim_like(uh, "zl"), dim_like(uh, "yh")]) / RHO0 / 1e6).mean())

def psi_window(f):
    ds = xr.open_dataset(f, decode_times=False)
    vmo = ds["vmo"].mean(dim=dim_like(ds["vmo"], "Time", "time"))
    rho = dim_like(vmo, "rho2", "rho", "zl")
    Psi = (vmo.sum(dim=dim_like(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=rho)
    return float(Psi.max()), float(Psi.min())

def En_over(rundir, wins):
    if not wins: return (np.nan, np.nan)
    lo, hi = int(wins[0]), int(wins[-1]) + 2000
    e = []
    for L in open(f"{rundir}/ocean.stats"):
        if ", En " not in L: continue
        p = L.split(",")
        try:
            d = float(p[1])
            if lo <= d < hi: e.append(float(p[3].split()[1]))
        except (IndexError, ValueError): pass
    e = np.array(e); return (e.mean(), e.std()) if len(e) else (np.nan, np.nan)

rows = {}
for name, (rundir, cutoff) in RUNS.items():
    wins = windows(rundir, cutoff)
    accs, pp, pm = [], [], []
    for w in wins:
        pt = f"{rundir}/output/prog_tmean_{w}.nc"; pr = f"{rundir}/output/prog_rho_tmean_{w}.nc"
        if not (glob.glob(pt) and glob.glob(pr)): continue
        accs.append(acc_window(pt)); a, b = psi_window(pr); pp.append(a); pm.append(b)
    en, ens = En_over(rundir, wins)
    rows[name] = dict(acc=np.mean(accs), acc_s=np.std(accs), pp=np.mean(pp), pp_s=np.std(pp),
                      pm=np.mean(pm), en=en, n=len(accs))

hdr = f"{'closure':<22}{'ACC [Sv]':>16}{'Psi_int [Sv]':>16}{'Psi_out [Sv]':>14}{'En [m2/s2]':>14}{'n':>3}"
print(hdr); print("-" * len(hdr))
for name, r in rows.items():
    print(f"{name:<22}{r['acc']:.4f}+/-{r['acc_s']:.4f} {r['pp']:>7.2f}+/-{r['pp_s']:<5.2f}{r['pm']:>12.2f}{r['en']:>14.3e}{r['n']:>3}")

# which MEKE khfac best matches truth interior overturning?
tgt = rows["truth 1/16 (p0625)"]["pp"]
print(f"\ntruth Psi_interior target = {tgt:.2f} Sv")
best = min([k for k in rows if k.startswith("MEKE")], key=lambda k: abs(rows[k]["pp"] - tgt))
for k in [x for x in rows if x.startswith("MEKE")]:
    print(f"  {k}: Psi_int {rows[k]['pp']:.2f}  (|diff| {abs(rows[k]['pp']-tgt):.2f}){'  <== closest' if k==best else ''}")
print(f"\nno-param vs ANN vs truth (KE, the affirmative axis):")
for k in ["truth 1/16 (p0625)", "no-param (1deg)", "ANN cb=1 (1deg)", best]:
    print(f"  {k:<22} En {rows[k]['en']:.3e}")

# --- ladder-vs-truth figure (A-diagnostics: interior overturning + ACC) ---
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order = ["no-param (1deg)", "MEKE khf=0.2", "MEKE khf=0.4", "MEKE khf=0.8", "ANN cb=1 (1deg)"]
xlab = ["no-param", "MEKE\n0.2", "MEKE\n0.4", "MEKE\n0.8", "ANN\n(cb=1)"]
col = ["#8c8c8c", "#9ecae1", "#4292c6", "#08519c", "#e6550d"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
panels = [("pp", "pp_s", r"Interior overturning $\Psi$"), ("acc", "acc_s", "ACC transport")]
for ax, (k, ks, ttl) in zip(axes, panels):
    ax.bar(range(len(order)), [rows[o][k] for o in order], yerr=[rows[o][ks] for o in order],
           color=col, capsize=3, edgecolor="k", linewidth=0.5)
    t, te = rows["truth 1/16 (p0625)"][k], rows["truth 1/16 (p0625)"][ks]
    ax.axhspan(t - te, t + te, color="k", alpha=0.12)
    ax.axhline(t, color="k", ls="--", lw=1.2, label=f"1/16° truth = {t:.2f} Sv")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(xlab)
    ax.set_ylabel(f"{ttl} [Sv]"); ax.set_title(ttl); ax.legend(loc="best", fontsize=9)
fig.suptitle("1° channel closures vs 1/16° truth (τ=0.2): ANN matches truth untuned; "
             "MEKE needs tuning and trades ACC for overturning", fontsize=10)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/meke_ladder_1deg.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print(f"\nwrote {png}")

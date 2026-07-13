"""1/4deg channel ladder vs 1/16 truth (Part 2 sec 4) — careful version.
Fixes: (1) EKE from SELF-CONSISTENT snapshot variance 0.5*<var_t(u)+var_t(v)> (>=0 by construction),
not En-minus-MKE; (2) inspect the full Psi(y,rho) field (a saved figure) rather than trusting max(Psi),
which the convergence notes flagged as finicky. Prints ACC, Psi(max & structure), resolved EKE."""
import xarray as xr, numpy as np, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO0 = 1035.0
B = "/scratch/db194/mom6/feb2026"
P25 = f"{B}/channel_extra_sponge_slow_woc_p25"
RUNS = {
    "truth 1/16":   (f"{B}/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0", 10000),
    "no-param":     (f"{P25}/tau_0.2_cb_0.0_cu_0.0",           4000),
    "ANN cb=1":     (f"{P25}/tau_0.2_cb_1.0_cu_0.0_neutral", 11500),
    "MEKE khf=0.4": (f"{P25}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500),
    "MEKE khf=0.8": (f"{P25}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.8", 11500),
}

def dl(da, *keys):
    for d in da.dims:
        if any(k == d or d.startswith(k) for k in keys): return d
    return None

def tmean_wins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]

def snap_files(rd, cut):
    fs = [f for f in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$", f)]
    return sorted(f for f in fs if int(f.split("_")[-1].split(".")[0]) >= cut)

def acc(f):
    uh = xr.open_dataset(f, decode_times=False)["uh"]
    return float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / RHO0 / 1e6).mean())

def psi_field(f):
    vmo = xr.open_dataset(f, decode_times=False)["vmo"]
    vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
    rho = dl(vmo, "rho2", "rho", "zl")
    Psi = (vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=rho)
    return Psi, vmo[dl(vmo, "yq")].values, xr.open_dataset(f, decode_times=False)[rho].values

def eke(rd, cut):  # resolved EKE = 0.5<var_t(u)+var_t(v)>, per snapshot window then averaged
    vals = []
    for f in snap_files(rd, cut):
        ds = xr.open_dataset(f, decode_times=False); u, v = ds["u"], ds["v"]; t = dl(u, "Time")
        vals.append(0.5 * (float(u.var(dim=t).mean()) + float(v.var(dim=t).mean())))
    return np.mean(vals) if vals else np.nan

rows, psifig = {}, {}
for name, (rd, cut) in RUNS.items():
    wins = tmean_wins(rd, cut)
    a = [acc(f"{rd}/output/prog_tmean_{w}.nc") for w in wins if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    prs = [f"{rd}/output/prog_rho_tmean_{w}.nc" for w in wins if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    Ps = [psi_field(f) for f in prs]
    Pmean = sum(p[0] for p in Ps) / len(Ps)
    psifig[name] = (Pmean, Ps[0][1], Ps[0][2])
    rows[name] = dict(acc=np.mean(a), acc_s=np.std(a), pmax=float(Pmean.max()), pmin=float(Pmean.min()),
                      eke=eke(rd, cut), n=len(a))

h = f"{'closure':<14}{'ACC[Sv]':>15}{'Psi_max':>9}{'Psi_min':>9}{'EKE(resolved)':>15}{'n':>3}"
print(h); print("-" * len(h))
for k, r in rows.items():
    print(f"{k:<14}{r['acc']:.4f}+/-{r['acc_s']:.4f}{r['pmax']:>9.2f}{r['pmin']:>9.2f}{r['eke']:>15.3e}{r['n']:>3}")
e0 = rows["no-param"]["eke"]
print("\nresolved EKE as % of no-param (least-damped):")
for k in ["no-param", "ANN cb=1", "MEKE khf=0.4", "MEKE khf=0.8"]:
    print(f"  {k:<14} {rows[k]['eke']/e0*100:5.0f}%")

# Psi(y,rho) panels to SEE the overturning structure across closures
fig, axes = plt.subplots(1, len(psifig), figsize=(4 * len(psifig), 4), sharey=True)
for ax, (nm, (P, yq, rho)) in zip(axes, psifig.items()):
    vmax = float(abs(P).max())
    pc = ax.pcolormesh(yq, rho, P.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"{nm}\nmax {float(P.max()):.1f} / min {float(P.min()):.1f} Sv", fontsize=9)
    ax.set_xlabel("lat"); ax.invert_yaxis(); plt.colorbar(pc, ax=ax)
axes[0].set_ylabel("rho2")
fig.suptitle("1/4deg residual overturning Psi(y,rho) by closure (tau=0.2) vs 1/16 truth", fontsize=10)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ladder_p25_psi.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"\nwrote {png}")

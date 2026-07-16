"""Bounded-Upsilon online matrix (1/4deg channel, cb=1, canonical neutral model, warm-start day 10000):
all four MESO_SFN_UPSILON_FORM options run side by side with the Upsilon-clamp OFF, plus one clamp-ON
rung, against the Jul-16 old-binary pair (clampoff 0.96 / deployment-clamped 0.84).

  Aoff_bnd  LOCAL_GRAD    clamp off  -> control: must reproduce old-binary clampoff (slope ~0.96)
  Boff      STENCIL_GRAD  clamp off  -> output-space bound via divisor floor (KEY QUESTION run)
  Coff      SLOPE_STENCIL clamp off  -> output-space bound via rescale-norm limit, slope inputs
  Bon       STENCIL_GRAD  clamp 15   -> is the safety clamp inert once the divisor is floored?
  Doff      SLOPE_CLAMP   clamp off  -> INPUT-space bound (Perezhogin): slope vector the ANN sees
                                        capped at slope_max. Doff vs Coff = input-vs-output bound.

Metrics per run: APE-removal slope + pattern r vs target (filtered 1/16 truth - no-param), resolved
EKE (ladder convention, % of no-param), ACC, Psi max/min. Same estimator as ape_eke_maps.py."""
import numpy as np, xarray as xr, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

RUNS = {  # name -> (subdir, cutoff)
    "old clampoff":  ("tau_0.2_cb_1.0_cu_0.0_clampoff", 11500),
    "old clamped":   ("tau_0.2_cb_1.0_cu_0.0_act_relu", 11500),
    "A LOCAL_GRAD":  ("tau_0.2_cb_1.0_cu_0.0_Aoff_bnd", 11500),
    "B STENCIL":     ("tau_0.2_cb_1.0_cu_0.0_Boff",     11500),
    "B STENCIL+cl":  ("tau_0.2_cb_1.0_cu_0.0_Bon",      11500),
    "C SLOPE_STEN":  ("tau_0.2_cb_1.0_cu_0.0_Coff",     11500),
    "D SLOPE_CLAMP": ("tau_0.2_cb_1.0_cu_0.0_Doff",     11500),
}

# --- target: what resolving eddies does to APE, on the 1/4deg grid ---
e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values; w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6

def fit(c):
    t, y = target.ravel(), c.ravel(); m = np.isfinite(t) & np.isfinite(y)
    return np.polyfit(t[m], y[m], 1)[0], np.corrcoef(t[m], y[m])[0, 1]

def snap_files(rd, cut):
    fs = [f for f in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$", f)]
    return sorted(f for f in fs if int(f.split("_")[-1].split(".")[0]) >= cut)

def eke(rd, cut):  # resolved EKE, ladder convention: 0.5<var_t(u)+var_t(v)> per snapshot file, averaged
    vals = []
    for f in snap_files(rd, cut):
        ds = xr.open_dataset(f, decode_times=False); u, v = ds["u"], ds["v"]; t = dl(u, "Time")
        vals.append(0.5 * (float(u.var(dim=t).mean()) + float(v.var(dim=t).mean()))); ds.close()
    return np.mean(vals) if vals else np.nan

def acc_psi(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    ws = [f"{w:06d}" for w in ws if w >= cut]
    a = []
    for w in ws:
        uh = xr.open_dataset(f"{rd}/output/prog_tmean_{w}.nc", decode_times=False)["uh"]
        a.append(float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / RHO0 / 1e6).mean()))
    Ps = []
    for w in ws:
        vmo = xr.open_dataset(f"{rd}/output/prog_rho_tmean_{w}.nc", decode_times=False)["vmo"]
        vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
        Ps.append((vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=dl(vmo, "rho2", "rho", "zl")))
    P = sum(Ps) / len(Ps)
    return np.mean(a), np.std(a), float(P.max()), float(P.min()), len(ws)

e0 = eke(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)
print(f"no-param resolved EKE reference: {e0:.3e}\n")

rows = {}
for nm, (sub, cut) in RUNS.items():
    rd = f"{P25}/{sub}"
    d = (ape_map(interfaces(rd, cut)[0], R, drho).values - A_np.values) / 1e6
    s, r = fit(d)
    a, a_s, pmax, pmin, n = acc_psi(rd, cut)
    rows[nm] = dict(slope=s, r=r, eke=eke(rd, cut), acc=a, acc_s=a_s, pmax=pmax, pmin=pmin, n=n, dmap=d)
    print(f"[{nm}] done (n={n} windows)")

h = f"\n{'run':<15}{'APEslope':>9}{'r':>7}{'EKE%np':>8}{'ACC[Sv]':>16}{'Psi_max':>9}{'Psi_min':>9}"
print(h); print("-" * len(h))
for nm, r in rows.items():
    print(f"{nm:<15}{r['slope']:>9.3f}{r['r']:>7.3f}{r['eke']/e0*100:>7.0f}%"
          f"{r['acc']:>10.4f}+/-{r['acc_s']:.4f}{r['pmax']:>9.2f}{r['pmin']:>9.2f}")

# clamp-inertness check: Bon vs Boff field-level agreement
db = rows["B STENCIL+cl"]["dmap"] - rows["B STENCIL"]["dmap"]
print(f"\nBon-Boff dAPE map: rms {np.nanstd(db):.3f} MJ/m2 (vs target rms {np.nanstd(target):.3f})"
      f" | EKE ratio Bon/Boff = {rows['B STENCIL+cl']['eke']/rows['B STENCIL']['eke']:.3f}")

# scatter figure: all four forms vs target
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharex=True, sharey=True)
t = target.ravel(); xl = np.array([np.nanmin(t), np.nanmax(t)])
for ax, nm in zip(axes, ["A LOCAL_GRAD", "B STENCIL", "C SLOPE_STEN", "D SLOPE_CLAMP"]):
    c = rows[nm]["dmap"].ravel(); m = np.isfinite(t) & np.isfinite(c); step = max(1, t[m].size // 4000)
    ax.scatter(t[m][::step], c[m][::step], s=5, alpha=0.25, color="#1a9850")
    ax.plot(xl, xl * rows[nm]["slope"], color="#1a9850", lw=2.2, label=f"slope {rows[nm]['slope']:.2f}, r {rows[nm]['r']:.2f}")
    ax.plot(xl, xl, "k--", lw=1); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f"{nm}  (EKE {rows[nm]['eke']/e0*100:.0f}% np)", fontsize=10)
    ax.set_xlabel("TARGET $\\Delta$APE [MJ m$^{-2}$]"); ax.legend(loc="upper left", fontsize=8)
axes[0].set_ylabel("closure $\\Delta$APE [MJ m$^{-2}$]")
fig.suptitle("Bounded-$\\Upsilon$ forms, $\\Upsilon$-clamp OFF: APE removal vs what resolving eddies does (1/4$^\\circ$, cb=1, neutral model)", fontsize=11)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/bounded_upsilon_matrix.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"\nwrote {png}")

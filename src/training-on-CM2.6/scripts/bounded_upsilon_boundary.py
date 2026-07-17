"""Boundary dissection of the bounded-Upsilon matrix: Pavel's division_by_zero.tex motivation was
APE BACKSCATTER (injection) along coastlines dominating the zonal-mean budget in CESM, with
MESO_UPSILON_CLAMP controlling even its sign. Question here: how much of that near-land effect
exists in our channel runs (walls at y=-50/-27, sponge -30.6 to -27.1), and do the bounded forms /
Upsilon-clamp act AT the walls or in the interior?

Diagnostics (state-based proxy for his budget maps): zonal-mean dAPE(y) = APE(run)-APE(no-param)
per run vs target; regularizer-action profiles (form - LOCAL_GRAD control, clampON - clampOFF);
wall-band concentration stats. Injection test: sign of dAPE near walls vs target."""
import numpy as np, xarray as xr, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

RUNS = {
    "A LOCAL_GRAD":  ("tau_0.2_cb_1.0_cu_0.0_Aoff_bnd", 11500),
    "B STENCIL":     ("tau_0.2_cb_1.0_cu_0.0_Boff",     11500),
    "C SLOPE_STEN":  ("tau_0.2_cb_1.0_cu_0.0_Coff",     11500),
    "D SLOPE_CLAMP": ("tau_0.2_cb_1.0_cu_0.0_Doff",     11500),
    "B +Uclamp":     ("tau_0.2_cb_1.0_cu_0.0_Bon",      11500),
    "old clampoff":  ("tau_0.2_cb_1.0_cu_0.0_clampoff", 11500),
    "old clamped":   ("tau_0.2_cb_1.0_cu_0.0_act_relu", 11500),
}
SPONGE_S, WALL_S, WALL_N = -30.625, -50.0, -27.0

e_tr, rho2 = interfaces(TRUTH, 10000)
lat16 = e_tr[dl(e_tr, "yh")].values; w16 = np.cos(np.deg2rad(lat16))
R = (e_tr * xr.DataArray(w16, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w16.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
yh = A_np[dl(A_np, "yh")].values; xh = A_np[dl(A_np, "xh")].values
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6
D = {nm: (ape_map(interfaces(f"{P25}/{sub}", cut)[0], R, drho).values - A_np.values) / 1e6
     for nm, (sub, cut) in RUNS.items()}

walldist = np.minimum(yh - WALL_S, WALL_N - yh)             # deg from nearest wall
def zm(f): return np.nanmean(f, axis=1)                     # zonal mean (y)
def band_stats(f, lo, hi):
    m = (yh >= lo) & (yh < hi)
    return np.nanmean(f[m, :])
def conc(f, d=1.5):  # fraction of area-integrated |f| within d deg of a wall (area frac of band printed for ref)
    wt = np.cos(np.deg2rad(yh))[:, None] * np.isfinite(f)
    a = np.nansum(np.abs(f) * wt); band = walldist < d
    return np.nansum(np.abs(f[band, :]) * wt[band, :]) / a, wt[band, :].sum() / wt.sum()

print(f"{'run':<14}{'S-wall band':>12}{'interior':>10}{'pre-sponge':>11}{'sponge':>9}   (zonal-mean dAPE, MJ/m2; bands: y<-48.5 | -48.5..-32 | -32..-30.6 | sponge)")
for nm, f in [("TARGET", target)] + list(D.items()):
    print(f"{nm:<14}{band_stats(f, -90, -48.5):>12.3f}{band_stats(f, -48.5, -32):>10.3f}"
          f"{band_stats(f, -32, SPONGE_S):>11.3f}{band_stats(f, SPONGE_S, 0):>9.3f}")

print("\nwall-concentration of the REGULARIZER ACTION (|effect| within 1.5deg of a wall):")
effects = {"bound B-A": D["B STENCIL"] - D["A LOCAL_GRAD"], "bound C-A": D["C SLOPE_STEN"] - D["A LOCAL_GRAD"],
           "bound D-A": D["D SLOPE_CLAMP"] - D["A LOCAL_GRAD"], "clamp Bon-Boff": D["B +Uclamp"] - D["B STENCIL"],
           "clamp old on-off": D["old clamped"] - D["old clampoff"]}
for nm, f in effects.items():
    c, af = conc(f)
    print(f"  {nm:<17} {c*100:5.1f}%  (band = {af*100:.1f}% of area)  rms {np.nanstd(f):.3f} MJ/m2")

# per-band slope vs target: does ANY form misbehave specifically near walls?
print("\nregression slope vs target, by band:")
for nm, f in D.items():
    ss = []
    for lo, hi in [(-90, -48.5), (-48.5, -32), (-32, SPONGE_S)]:
        m = (yh >= lo) & (yh < hi); t, y = target[m, :].ravel(), f[m, :].ravel()
        k = np.isfinite(t) & np.isfinite(y)
        ss.append(np.polyfit(t[k], y[k], 1)[0])
    print(f"  {nm:<14} S-wall {ss[0]:5.2f}  interior {ss[1]:5.2f}  pre-sponge {ss[2]:5.2f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
ax.plot(yh, zm(target), "k", lw=2.5, label="TARGET (truth - no-param)")
for nm, c_ in [("A LOCAL_GRAD", "#1a9850"), ("B STENCIL", "#66bd63"), ("C SLOPE_STEN", "#a6d96a"), ("D SLOPE_CLAMP", "#3288bd")]:
    ax.plot(yh, zm(D[nm]), color=c_, lw=1.4, label=nm)
ax.plot(yh, zm(D["B +Uclamp"]), color="#d73027", lw=1.4, ls="--", label="B +Uclamp")
ax.plot(yh, zm(D["old clamped"]), color="#7b3294", lw=1.2, ls=":", label="old clamped")
ax.axvspan(SPONGE_S, WALL_N, color="gray", alpha=0.18, label="sponge")
ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("lat"); ax.set_ylabel("zonal-mean $\\Delta$APE [MJ m$^{-2}$]")
ax.set_title("APE change vs no-param: ALL forms over-inject at the S wall\n(right sign, ~2$\\times$ target amplitude)", fontsize=10)
ax.legend(fontsize=7, loc="lower right")

ax = axes[1]
for nm, c_ in [("bound B-A", "#66bd63"), ("bound C-A", "#a6d96a"), ("bound D-A", "#3288bd")]:
    ax.plot(yh, zm(effects[nm]), color=c_, lw=1.4, label=nm)
ax.plot(yh, zm(effects["clamp Bon-Boff"]), color="#d73027", lw=1.6, ls="--", label="clamp Bon-Boff")
ax.plot(yh, zm(effects["clamp old on-off"]), color="#7b3294", lw=1.4, ls=":", label="clamp old on-off")
ax.axvspan(SPONGE_S, WALL_N, color="gray", alpha=0.18)
ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("lat")
ax.set_title("Where the regularizers act (zonal-mean effect)", fontsize=10); ax.legend(fontsize=8)

ax = axes[2]
vm = np.nanpercentile(np.abs(effects["clamp Bon-Boff"]), 99)
pc = ax.pcolormesh(xh, yh, effects["clamp Bon-Boff"], cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
ax.axhspan(SPONGE_S, WALL_N, color="gray", alpha=0.25)
plt.colorbar(pc, ax=ax, label="MJ m$^{-2}$"); ax.set_xlabel("lon"); ax.set_ylabel("lat")
ax.set_title("Clamp action map (Bon $-$ Boff $\\Delta$APE)", fontsize=10)
fig.suptitle("Is Pavel's coastline-APE-injection effect present in the channel? (1/4$^\\circ$, cb=1, neutral model)", fontsize=11)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/bounded_upsilon_boundary.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"\nwrote {png}")

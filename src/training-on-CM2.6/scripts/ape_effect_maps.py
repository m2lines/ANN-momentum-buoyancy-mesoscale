"""'Show the effect' figure (Pavel-Fig-2 spirit): at 1/4deg, the change in depth-integrated APE that each
closure produces, next to the target = what resolving eddies (filtered truth) does. Top row: three ΔAPE maps
(target / ANN-canonical / MEKE), shared scale. Bottom: scatter of each closure's ΔAPE vs the target, which
makes the regression 'slope' literal (ANN ~0.9 = ~90% of the target amplitude; MEKE ~1.2 = overshoots)."""
import numpy as np, xarray as xr, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# reuse the estimator machinery (dl, interfaces, block, ape_map, constants B, TRUTH, P25)
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

# common reference = truth mean interfaces; target at 1/4deg
e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values; w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_trf = ape_map(block(e_tr, 4), R, drho)
e_np, _ = interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)
A_np = ape_map(e_np, R, drho)
target = (A_trf.values - A_np.values) / 1e6           # MJ/m2

def dmap(rundir, cut):
    e, _ = interfaces(rundir, cut)
    return (ape_map(e, R, drho).values - A_np.values) / 1e6

ann  = dmap(f"{P25}/tau_0.2_cb_1.0_cu_0.0_clampoff", 11500)   # unclamped Upsilon (new binary, clamps disabled)
meke = dmap(f"{P25}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500)
xh = A_np[dl(A_np, "xh")].values; yh = A_np[dl(A_np, "yh")].values

def fit(c):
    t, y = target.ravel(), c.ravel(); m = np.isfinite(t) & np.isfinite(y)
    s = np.polyfit(t[m], y[m], 1)[0]; r = np.corrcoef(t[m], y[m])[0, 1]; return s, r

s_ann, r_ann = fit(ann); s_mk, r_mk = fit(meke)
vm = np.nanpercentile(np.abs(target), 99)

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.18)
panels = [("what resolving eddies does\n(truth $-$ no-param) = TARGET", target, None),
          (f"ANN (unclamped $\\Upsilon$)\nslope {s_ann:.2f}, pattern r {r_ann:.2f}", ann, s_ann),
          (f"MEKE\nslope {s_mk:.2f}, pattern r {r_mk:.2f}", meke, s_mk)]
for j, (ttl, fld, sl) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    pc = ax.pcolormesh(xh, yh, fld, cmap="PuOr_r", vmin=-vm, vmax=vm, shading="auto")
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("lon")
    if j == 0: ax.set_ylabel("lat")
    plt.colorbar(pc, ax=ax, fraction=0.046, label="$\\Delta$APE [MJ m$^{-2}$]" if j == 2 else "")

# scatter: closure ΔAPE vs target, the slope made literal
ax = fig.add_subplot(gs[1, :])
t = target.ravel(); mA = np.isfinite(t) & np.isfinite(ann.ravel()); mM = np.isfinite(t) & np.isfinite(meke.ravel())
rng = np.random.default_rng  # (unused; keep sampling deterministic via slicing)
step = max(1, t[mA].size // 4000)
ax.scatter(t[mA][::step], ann.ravel()[mA][::step], s=6, alpha=0.25, color="#e6550d", label=f"ANN  (slope {s_ann:.2f})")
ax.scatter(t[mM][::step], meke.ravel()[mM][::step], s=6, alpha=0.25, color="#3182bd", label=f"MEKE (slope {s_mk:.2f})")
xline = np.array([np.nanmin(t), np.nanmax(t)])
ax.plot(xline, xline, "k--", lw=1, label="1:1 (perfect: removes what eddies do)")
ax.plot(xline, s_ann * xline, color="#e6550d", lw=2)
ax.plot(xline, s_mk * xline, color="#3182bd", lw=2)
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("TARGET $\\Delta$APE per point  [MJ m$^{-2}$]  (what resolving eddies does)")
ax.set_ylabel("closure $\\Delta$APE  [MJ m$^{-2}$]")
ax.set_title("Each point = one location. Slope = fraction of the eddies' APE change the closure reproduces.", fontsize=10)
ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Does the closure change the ocean's APE the way resolving eddies would?  (1/4°, $\\Delta$ relative to no-param)", fontsize=12)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_effect_maps_unclamped.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print("wrote", png, "| ANN slope %.2f r %.2f | MEKE slope %.2f r %.2f" % (s_ann, r_ann, s_mk, r_mk))

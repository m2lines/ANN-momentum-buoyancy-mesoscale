"""Side-by-side: the Upsilon-clamp's effect on the ANN's APE removal (1/4deg). Same binary + neutral model,
clamp the only difference. Top: target / ANN clamp-OFF / ANN clamp-ON (deployment) ΔAPE maps. Bottom: both
ANN scatter clouds vs target with regression lines — the clamp pulls the slope from 0.96 down to 0.84."""
import numpy as np, xarray as xr, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values; w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6
def dmap(sub, cut=11500):
    return (ape_map(interfaces(f"{P25}/{sub}", cut)[0], R, drho).values - A_np.values) / 1e6
off = dmap("tau_0.2_cb_1.0_cu_0.0_clampoff")      # unclamped
on  = dmap("tau_0.2_cb_1.0_cu_0.0_act_relu")      # clamped (deployment)
xh = A_np[dl(A_np, "xh")].values; yh = A_np[dl(A_np, "yh")].values
def fit(c):
    t, y = target.ravel(), c.ravel(); m = np.isfinite(t) & np.isfinite(y); return np.polyfit(t[m], y[m], 1)[0], np.corrcoef(t[m], y[m])[0, 1]
s_off, r_off = fit(off); s_on, r_on = fit(on)
vm = np.nanpercentile(np.abs(target), 99)

fig = plt.figure(figsize=(13, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.18)
for j, (ttl, fld) in enumerate([("what resolving eddies does\n(truth $-$ no-param) = TARGET", target),
                                (f"ANN, $\\Upsilon$ UNCLAMPED\nslope {s_off:.2f}", off),
                                (f"ANN, $\\Upsilon$ CLAMPED (deployment)\nslope {s_on:.2f}", on)]):
    ax = fig.add_subplot(gs[0, j])
    pc = ax.pcolormesh(xh, yh, fld, cmap="PuOr_r", vmin=-vm, vmax=vm, shading="auto")
    ax.set_title(ttl, fontsize=10); ax.set_xlabel("lon"); ax.set_ylabel("lat" if j == 0 else "")
    plt.colorbar(pc, ax=ax, fraction=0.046, label="$\\Delta$APE [MJ m$^{-2}$]" if j == 2 else "")

ax = fig.add_subplot(gs[1, :])
t = target.ravel()
for c, col, lbl, s in [(off, "#1a9850", f"unclamped (slope {s_off:.2f})", s_off), (on, "#d73027", f"clamped/deployment (slope {s_on:.2f})", s_on)]:
    m = np.isfinite(t) & np.isfinite(c.ravel()); step = max(1, t[m].size // 4000)
    ax.scatter(t[m][::step], c.ravel()[m][::step], s=6, alpha=0.2, color=col)
    ax.plot([np.nanmin(t), np.nanmax(t)], np.array([np.nanmin(t), np.nanmax(t)]) * s, color=col, lw=2.4, label=lbl)
xl = np.array([np.nanmin(t), np.nanmax(t)]); ax.plot(xl, xl, "k--", lw=1, label="1:1 (removes exactly what eddies do)")
ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("TARGET $\\Delta$APE per point [MJ m$^{-2}$] (what resolving eddies does)")
ax.set_ylabel("ANN $\\Delta$APE [MJ m$^{-2}$]")
ax.set_title("The $\\Upsilon$-clamp caps the flux in weak-gradient regions $\\Rightarrow$ pulls the APE-removal slope from 0.96 to 0.84", fontsize=10)
ax.legend(loc="upper left", fontsize=9)
fig.suptitle("Effect of the $\\Upsilon$-clamp (division-by-zero fix) on the ANN's APE removal — 1/4°, same binary & model", fontsize=12)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/clamp_ape_compare.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print("wrote", png, "| unclamped %.2f  clamped %.2f" % (s_off, s_on))

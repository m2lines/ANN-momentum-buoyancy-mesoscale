"""Cleaner presentation of the boundary story (Dhruv's request): every choice shown in the SAME
currency. Top row: dAPE = (run - no-param) for TRUTH, unbounded control, bounded form, bounded+clamp,
all on one color scale. Bottom row: error = (run - TRUTH dAPE) for the three ANN runs, one scale.
No run-minus-run panels. Bathymetry panel anchors the geometry (S wall + ridge at lon 20)."""
import numpy as np, xarray as xr, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

SPONGE_S, WALL_N = -30.625, -27.0
e_tr, rho2 = interfaces(TRUTH, 10000)
lat16 = e_tr[dl(e_tr, "yh")].values; w16 = np.cos(np.deg2rad(lat16))
R = (e_tr * xr.DataArray(w16, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w16.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
yh = A_np[dl(A_np, "yh")].values; xh = A_np[dl(A_np, "xh")].values
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6

def dmap_and_depth(sub, cut=11500):
    e, _ = interfaces(f"{P25}/{sub}", cut)
    rho = dl(e, "rho2", "rho")
    return (ape_map(e, R, drho).values - A_np.values) / 1e6, e.isel({rho: -1}).values

A, depth = dmap_and_depth("tau_0.2_cb_1.0_cu_0.0_Aoff_bnd")
C, _ = dmap_and_depth("tau_0.2_cb_1.0_cu_0.0_Coff")
Bon, _ = dmap_and_depth("tau_0.2_cb_1.0_cu_0.0_Bon")

wall = yh < -48.5                                   # southern-wall band
def wmean(f): return np.nanmean(f[wall, :])

fig, ax = plt.subplots(2, 4, figsize=(21, 8.4))
for a in list(ax[0, 1:]) + list(ax[1, 1:]):
    a.sharex(ax[0, 1]); a.sharey(ax[0, 1])
def topo(a, lw=0.7):
    a.contour(xh, yh, depth, levels=[1750, 2250, 2750], colors="k", linewidths=lw, alpha=0.6)
    a.axhspan(SPONGE_S, WALL_N, color="gray", alpha=0.25)
    a.axhline(-48.5, color="k", ls=":", lw=0.9)

pc = ax[0, 0].pcolormesh(xh, yh, depth, cmap="Blues", shading="auto")
plt.colorbar(pc, ax=ax[0, 0], label="m"); topo(ax[0, 0])
ax[0, 0].set_title("bathymetry: ridge crest 1500 m of 3000 m\ngray = sponge, dotted = S-wall band", fontsize=10)
ax[0, 0].set_ylabel("lat")

vm = np.nanpercentile(np.abs(target), 99)
for a, fld, ttl in [(ax[0, 1], target, "TRUTH $-$ no-param  (the target)"),
                    (ax[0, 2], A, "unbounded ANN $-$ no-param  (LOCAL_GRAD)"),
                    (ax[0, 3], C, "bounded ANN $-$ no-param  (SLOPE_STENCIL)")]:
    pc = a.pcolormesh(xh, yh, fld, cmap="PuOr_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=a, label="MJ m$^{-2}$"); topo(a); a.set_title(ttl, fontsize=10)
    a.text(0.02, 0.03, f"S-wall band {wmean(fld):+.2f}", transform=a.transAxes, fontsize=9,
           bbox=dict(fc="w", ec="0.6", alpha=0.85, pad=1.5))

ax[1, 0].plot(np.nanmean(target, axis=1), yh, "k", lw=2.5, label="TRUTH (target)")
for fld, c_, lab in [(A, "#1a9850", "unbounded"), (C, "#3288bd", "bounded"), (Bon, "#d73027", "bounded+$\\Upsilon$clamp")]:
    ax[1, 0].plot(np.nanmean(fld, axis=1), yh, color=c_, lw=1.5, label=lab)
ax[1, 0].axhspan(SPONGE_S, WALL_N, color="gray", alpha=0.25); ax[1, 0].axhline(-48.5, color="k", ls=":", lw=0.9)
ax[1, 0].axvline(0, color="gray", lw=0.5); ax[1, 0].set_ylim(yh[0], yh[-1])
ax[1, 0].set_xlabel("zonal-mean $\\Delta$APE [MJ m$^{-2}$]"); ax[1, 0].set_ylabel("lat")
ax[1, 0].set_title("zonal means: all runs over-shoot at the wall", fontsize=10)
ax[1, 0].legend(fontsize=8, loc="upper left")

vme = 1.2
for a, fld, ttl in [(ax[1, 1], A - target, "error: unbounded $-$ TRUTH"),
                    (ax[1, 2], C - target, "error: bounded $-$ TRUTH"),
                    (ax[1, 3], Bon - target, "error: bounded$+\\Upsilon$clamp $-$ TRUTH")]:
    pc = a.pcolormesh(xh, yh, fld, cmap="RdBu_r", vmin=-vme, vmax=vme, shading="auto")
    plt.colorbar(pc, ax=a, label="MJ m$^{-2}$"); topo(a); a.set_title(ttl, fontsize=10)
    a.text(0.02, 0.03, f"S-wall band {wmean(fld):+.2f}   domain rms {np.nanstd(fld):.2f}",
           transform=a.transAxes, fontsize=9, bbox=dict(fc="w", ec="0.6", alpha=0.85, pad=1.5))

for a in ax[1, 1:]: a.set_xlabel("lon")
ax[0, 0].set_xlabel("lon")
fig.suptitle("Top: what each choice does to APE (run $-$ no-param, one color scale). "
             "Bottom: how each misses the truth (run $-$ target).  1/4$^\\circ$, cb=1, neutral model",
             fontsize=12)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/bounded_upsilon_maps_v2.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"wrote {png}")

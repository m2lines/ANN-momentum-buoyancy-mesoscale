"""Make the boundary/topography story visual: maps of the wall over-injection, the bounded-form fix,
and the Upsilon-clamp action, all over the channel's actual bathymetry (southern wall + the meridional
Gaussian ridge at lon 20, crest 1500 m of 3000 m, TOPO_CONFIG=seamount X_L=2.25deg Y_L=0).
Depth is data-true: bottom interface = column sum of time-mean thkcello (Boussinesq).
Also prints ridge-strip vs off-ridge stats for excess / bound action / clamp action."""
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
Boff, _ = dmap_and_depth("tau_0.2_cb_1.0_cu_0.0_Boff")
excess, bound_act, clamp_act = A - target, C - A, Bon - Boff

# ridge-strip vs off-ridge stats (interior lats only, away from wall band and sponge)
inter = (yh >= -48.5) & (yh < -32)
ridge = np.abs(xh - 20) <= 3
for nm, f in [("excess A-target", excess), ("bound C-A", bound_act), ("clamp Bon-Boff", clamp_act)]:
    on = np.nanmean(f[np.ix_(inter, ridge)]); off = np.nanmean(f[np.ix_(inter, ~ridge)])
    print(f"{nm:<16} interior ridge-strip {on:+.3f}  off-ridge {off:+.3f} MJ/m2")
print(f"wall band (y<-48.5): excess {np.nanmean(excess[yh < -48.5, :]):+.3f}  bound {np.nanmean(bound_act[yh < -48.5, :]):+.3f}  clamp {np.nanmean(clamp_act[yh < -48.5, :]):+.3f}")

fig, ax = plt.subplots(2, 3, figsize=(16.5, 8), sharex=True, sharey=True)
def topo(a, lw=0.7):
    a.contour(xh, yh, depth, levels=[1750, 2250, 2750], colors="k", linewidths=lw, alpha=0.6)
    a.axhspan(SPONGE_S, WALL_N, color="gray", alpha=0.25)

pc = ax[0, 0].pcolormesh(xh, yh, depth, cmap="Blues", shading="auto")
plt.colorbar(pc, ax=ax[0, 0], label="m"); topo(ax[0, 0])
ax[0, 0].set_title("bathymetry: S wall + meridional ridge (crest 1500 m)\ngray = N sponge; contours 1750/2250/2750 m", fontsize=10)

vm = np.nanpercentile(np.abs(target), 99)
for a, fld, ttl in [(ax[0, 1], target, "TARGET $\\Delta$APE (truth $-$ no-param)"),
                    (ax[0, 2], A, "ANN LOCAL_GRAD $\\Delta$APE (unbounded control)")]:
    pc = a.pcolormesh(xh, yh, fld, cmap="PuOr_r", vmin=-vm, vmax=vm, shading="auto")
    plt.colorbar(pc, ax=a, label="MJ m$^{-2}$"); topo(a); a.set_title(ttl, fontsize=10)

for a, fld, vmx, ttl in [
        (ax[1, 0], excess, 1.2, "over-injection: ANN $-$ TARGET\n(positive band pinned to the S wall)"),
        (ax[1, 1], bound_act, 0.25, "bounded-form fix: SLOPE_STENCIL $-$ LOCAL_GRAD\n(removes APE right at the wall)"),
        (ax[1, 2], clamp_act, 0.25, "$\\Upsilon$-clamp action: Bon $-$ Boff\n(acts over the ridge + eddy-active interior)")]:
    pc = a.pcolormesh(xh, yh, fld, cmap="RdBu_r", vmin=-vmx, vmax=vmx, shading="auto")
    plt.colorbar(pc, ax=a, label="MJ m$^{-2}$"); topo(a); a.set_title(ttl, fontsize=10)

for a in ax[1]: a.set_xlabel("lon")
for a in ax[:, 0]: a.set_ylabel("lat")
fig.suptitle("Where the near-land $\\Upsilon$ pathology lives in the channel (1/4$^\\circ$, cb=1, neutral model):\n"
             "southern wall vs ridge-at-depth vs interior", fontsize=12)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/bounded_upsilon_boundary_maps.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"wrote {png}")

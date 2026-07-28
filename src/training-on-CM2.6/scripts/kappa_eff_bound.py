"""Is the 'bounded' Upsilon actually bounded? Dhruv's point: in Psi = dx*dy*||grad u||*||s||*ANN,
bounding the slope bounds only the s factor -- the effective diffusivity kappa_eff = dx*dy*||grad u||*C
is a FLOW quantity with no a priori ceiling. Pavel's |Psi|<=10 m2/s rests on GM's kappa<=1000, which
is a value one SETS, not a structural bound.

Computes kappa_eff = dx*dy*||[grad u]_3x3||_2 (the ANN's own 'diffusivity' prefactor) from the 1/4deg
runs, and asks: (a) how big is it vs GM's 1000 m2/s, (b) how heavy is its tail, (c) does the
Upsilon-CLAMP's action map co-locate with the large-kappa_eff tail -- i.e. is the clamp doing the job
the slope bound structurally cannot?"""
import numpy as np, xarray as xr, glob, re
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])
RE = 6.371e6

def grad_u_norm(u, v, lat, dxdeg=0.25):
    """||[grad u]_3x3||_2 as the code builds it: Frobenius norm of (du/dx,du/dy,dv/dx,dv/dy) over a
    3x3 stencil. Computed here as the local norm times 3 (sqrt(9) points), which is the convention
    the stencil-norm factor of 3 encodes."""
    dy = np.deg2rad(dxdeg) * RE
    dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    ux = np.gradient(u, axis=-1) / dx; uy = np.gradient(u, axis=-2) / dy
    vx = np.gradient(v, axis=-1) / dx; vy = np.gradient(v, axis=-2) / dy
    return 3.0 * np.sqrt(ux**2 + uy**2 + vx**2 + vy**2), dx, dy

def kappa_eff(sub, cut=11500, nfile=2):
    """kappa_eff = dx*dy*||[grad u]||_2 from instantaneous snapshots (what the closure actually sees)."""
    out = []
    fs = [f for f in sorted(glob.glob(f"{P25}/{sub}/output/prog_*.nc")) if re.match(r".*/prog_\d+\.nc$", f)
          and int(f.split("_")[-1].split(".")[0]) >= cut][:nfile]
    for f in fs:
        ds = xr.open_dataset(f, decode_times=False, chunks={"Time": 2})
        t = dl(ds, "Time", "time")
        xq, yq = dl(ds, "xq"), dl(ds, "yq")
        u = 0.5 * (ds["u"].isel({xq: slice(0, -1)}) + ds["u"].isel({xq: slice(1, None)}))
        v = 0.5 * (ds["v"].isel({yq: slice(0, -1)}) + ds["v"].isel({yq: slice(1, None)}))
        lat = ds[dl(ds, "yh")].values
        for it in range(ds.sizes[t]):
            gu, dx, dy = grad_u_norm(u.isel({t: it}).values, v.isel({t: it}).values, lat)
            out.append(dx * dy * gu)                            # (z,y,x), m2/s
        ds.close()
    return np.concatenate([o[None] for o in out], axis=0), lat   # (n,z,y,x)

K, lat = kappa_eff("tau_0.2_cb_1.0_cu_0.0_Boff")
Kf = K[np.isfinite(K)]
print(f"kappa_eff = dx*dy*||[grad u]_3x3||  (1/4deg, STENCIL_GRAD run, {K.shape[0]} snapshots)")
for p in [50, 90, 99, 99.9, 100]:
    print(f"   {p:>5.1f}th pct : {np.percentile(Kf, p):10.1f} m2/s")
print(f"   fraction exceeding GM's 1000 m2/s : {100*np.mean(Kf > 1000):.1f}%")
print(f"   fraction exceeding 10x that       : {100*np.mean(Kf > 1e4):.2f}%")
print(f"   implied |Psi| <= kappa_eff * s_max(0.01) : median {0.01*np.median(Kf):.2f}, "
      f"99.9th {0.01*np.percentile(Kf, 99.9):.1f} m2/s   (clamp is 15)")

# does the clamp act where kappa_eff is large?
e_tr, rho2 = interfaces(TRUTH, 10000)
lat16 = e_tr[dl(e_tr, "yh")].values; w16 = np.cos(np.deg2rad(lat16))
R = (e_tr * xr.DataArray(w16, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w16.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
def dm(sub): return (ape_map(interfaces(f"{P25}/{sub}", 11500)[0], R, drho).values - A_np.values) / 1e6
clamp_act = dm("tau_0.2_cb_1.0_cu_0.0_Bon") - dm("tau_0.2_cb_1.0_cu_0.0_Boff")
bound_act = dm("tau_0.2_cb_1.0_cu_0.0_Coff") - dm("tau_0.2_cb_1.0_cu_0.0_Aoff_bnd")
Kcol = np.nanmean(K, axis=(0, 1))                                # time+depth mean map
for nm, f in [("clamp Bon-Boff", clamp_act), ("bound C-A", bound_act)]:
    a, b = Kcol.ravel(), np.abs(f).ravel(); k = np.isfinite(a) & np.isfinite(b)
    print(f"\ncorr(|{nm}|, kappa_eff) = {np.corrcoef(a[k], b[k])[0,1]:+.3f}")
    hi = a >= np.nanpercentile(a[k], 75)
    print(f"   mean |effect| in top-quartile kappa_eff {np.nanmean(b[hi & k]):.4f} "
          f"vs rest {np.nanmean(b[~hi & k]):.4f}  (ratio {np.nanmean(b[hi&k])/np.nanmean(b[~hi&k]):.2f})")

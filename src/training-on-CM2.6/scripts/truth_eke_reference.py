"""Resolved-EKE reference the cb question needs: is the ANN run EKE-deficient or EKE-excessive
relative to the truth? Existing ladders quote EKE only as % of no-param, which cannot answer that.

Fair comparison: velocities interpolated to cell centres, truth block-averaged 4x to the 1/4deg
grid FIRST, then time variance -- i.e. the EKE a 1/4deg grid is supposed to carry.
EKE = 0.5*<var_t(u_c) + var_t(v_c)>, area-weighted, all levels, per snapshot file then averaged."""
import numpy as np, xarray as xr, glob, re
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

def centred(ds):
    u, v = ds["u"], ds["v"]
    xq, yq = dl(u, "xq"), dl(v, "yq")
    uc = 0.5 * (u.isel({xq: slice(0, -1)}).values + u.isel({xq: slice(1, None)}).values)
    vc = 0.5 * (v.isel({yq: slice(0, -1)}).values + v.isel({yq: slice(1, None)}).values)
    return uc, vc                                             # (t,z,y,x) on cell centres

def blk(a, n):                                                # block-mean last two axes
    if n == 1: return a
    t, z, y, x = a.shape
    return a.reshape(t, z, y // n, n, x // n, n).mean(axis=(3, 5))

def eke_cg(rundir, cut, n=1, lat=None):
    tot, m = 0.0, 0
    for f in sorted(glob.glob(f"{rundir}/output/prog_*.nc")):
        if not re.match(r".*/prog_\d+\.nc$", f): continue
        if int(f.split("_")[-1].split(".")[0]) < cut: continue
        ds = xr.open_dataset(f, decode_times=False)
        if ds.sizes[dl(ds, "Time", "time")] < 2: ds.close(); continue
        uc, vc = centred(ds)
        yv = ds[dl(ds, "yh")].values
        ds.close()
        uc, vc = blk(uc, n), blk(vc, n)
        yv = yv[: (len(yv) // n) * n].reshape(-1, n).mean(axis=1) if n > 1 else yv
        k = 0.5 * (np.nanvar(uc, axis=0) + np.nanvar(vc, axis=0))          # (z,y,x)
        w = np.cos(np.deg2rad(yv))[None, :, None] * np.ones_like(k)
        w[~np.isfinite(k)] = np.nan
        tot += float(np.nansum(k * w) / np.nansum(w)); m += 1
        print(f"    {f.split('/')[-1]}  EKE {tot/m:.4e}", flush=True)
    return tot / max(m, 1)

print("TRUTH 1/16deg block-averaged to 1/4deg:", flush=True)
e_truth = eke_cg(TRUTH, 10000, n=4)
print("no-param 1/4deg:", flush=True)
e_np = eke_cg(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)
print(f"\n{'run':<16}{'EKE':>12}{'% of truth':>12}{'% of no-param':>15}")
print(f"{'TRUTH (cg)':<16}{e_truth:>12.4e}{100:>12.0f}{100*e_truth/e_np:>15.0f}")
print(f"{'no-param':<16}{e_np:>12.4e}{100*e_np/e_truth:>12.0f}{100:>15.0f}")
for nm, sub in [("ANN cb=1", "tau_0.2_cb_1.0_cu_0.0_neutral"), ("ANN cb=2", "tau_0.2_cb_2.0_cu_0.0_neutral"),
                ("ANN cb=3", "tau_0.2_cb_3.0_cu_0.0_neutral"), ("ANN cb=4", "tau_0.2_cb_4.0_cu_0.0_neutral")]:
    e = eke_cg(f"{P25}/{sub}", 4000)
    print(f"{nm:<16}{e:>12.4e}{100*e/e_truth:>12.0f}{100*e/e_np:>15.0f}", flush=True)

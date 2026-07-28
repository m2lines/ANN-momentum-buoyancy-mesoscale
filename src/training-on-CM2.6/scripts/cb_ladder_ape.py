"""Does turning the ANN's scaling coefficient up fix the APE under-drainage?
Scores the existing cb ladder (cb=1,2,3,4, neutral model, tau=0.2, cu=0) with the same
APE-fidelity estimator used for the bounded-Upsilon matrix: regression slope + pattern r of
dAPE(run - no-param) on target(truth - no-param), plus resolved EKE as % of no-param, plus the
S-wall band mean (does cranking cb amplify the near-wall over-injection?)."""
import numpy as np, xarray as xr, glob, re
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

RUNS = [("cb=1 neutral", "tau_0.2_cb_1.0_cu_0.0_neutral"), ("cb=2 neutral", "tau_0.2_cb_2.0_cu_0.0_neutral"),
        ("cb=3 neutral", "tau_0.2_cb_3.0_cu_0.0_neutral"), ("cb=4 neutral", "tau_0.2_cb_4.0_cu_0.0_neutral")]

DAY = re.compile(r"_(\d+)\.nc")
for nm, sub in RUNS:                                     # what time range do these cover?
    f = sorted(glob.glob(f"{P25}/{sub}/output/prog_tmean_*.nc"))
    days = [DAY.search(x).group(1) for x in f]
    print(f"{nm:<14} {len(f)} tmean files: {days}")

e_tr, rho2 = interfaces(TRUTH, 10000)
lat16 = e_tr[dl(e_tr, "yh")].values; w16 = np.cos(np.deg2rad(lat16))
R = (e_tr * xr.DataArray(w16, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w16.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
yh = A_np[dl(A_np, "yh")].values
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6
wall = yh < -48.5

def eke(sub, cut):
    """Resolved EKE = 0.5<var_t(u)+var_t(v)> per snapshot file, averaged over files.
    Chunked over Time so a full prog file never lands in memory at once."""
    tot, n = 0.0, 0
    for f in sorted(glob.glob(f"{sub}/output/prog_*.nc")):
        if int(re.search(r"prog_(\d+)\.nc", f).group(1)) < cut: continue
        d = xr.open_dataset(f, decode_times=False, chunks={"Time": 4})
        t = dl(d, "Time", "time")
        if d.sizes[t] < 2: d.close(); continue                  # var undefined on a single record
        k = 0.5 * (d.u.var(dim=t) + d.v.var(dim=t))
        w = np.cos(np.deg2rad(k[dl(k, "yh")].values)) if dl(k, "yh") in k.dims else 1.0
        tot += float((k * xr.DataArray(w, dims=dl(k, "yh"))).mean().compute() / np.mean(w)); n += 1
        d.close()
    return tot / max(n, 1)

E_NP = eke(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)
print(f"\nno-param EKE = {E_NP:.4e}\n")
print(f"{'run':<14}{'slope':>8}{'r':>7}{'EKE %np':>9}{'S-wall dAPE':>13}{'wall err':>10}{'interior err':>14}")
print(f"{'TARGET':<14}{1.000:>8.3f}{1.000:>7.3f}{'--':>9}{np.nanmean(target[wall,:]):>13.3f}{0.0:>10.3f}{0.0:>14.3f}")
inter = (yh >= -48.5) & (yh < -32)
for nm, sub in RUNS:
    cut = 4000
    f = (ape_map(interfaces(f"{P25}/{sub}", cut)[0], R, drho).values - A_np.values) / 1e6
    t, y = target.ravel(), f.ravel(); k = np.isfinite(t) & np.isfinite(y)
    s = np.polyfit(t[k], y[k], 1)[0]; r = np.corrcoef(t[k], y[k])[0, 1]
    e = eke(f"{P25}/{sub}", cut)
    print(f"{nm:<14}{s:>8.3f}{r:>7.3f}{100*e/E_NP:>9.1f}{np.nanmean(f[wall,:]):>13.3f}"
          f"{np.nanmean((f-target)[wall,:]):>10.3f}{np.nanmean((f-target)[inter,:]):>14.3f}")
print("\n(slope 1 = right amplitude; S-wall dAPE target is +0.24; wall/interior err = run - target)")

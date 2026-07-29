"""The model's OWN answer, from the 100-day diagnostic re-runs (Boff_diag = STENCIL_GRAD clamp off,
Bon_diag = STENCIL_GRAD clamp on). Delivers two things the offline reconstruction could not:

  1. the WOULD-FIRE fraction -- |Upsilon| > 15 from daily INSTANTANEOUS Upsilon snapshots under the
     bounded form with the clamp inactive (the deliverable of the Jul-16 matrix task that was never
     produced, because those runs' diag_table requested only meso_sfn_drdy_{c,v});
  2. the ACTUAL activation frequency -- the time-mean of the model's own Upsilon_clamp_{u,v} mask in
     the clamped run, which is what the clamp really does in the deployment configuration.

It also checks the bounded form's own guarantee, s_hat <= 3*slope_max = 0.03, and compares the
realized streamfunction against the unlimited one (GM_sfn vs GM_sfn_unlim) to see how often MOM6's
downstream mass limiter binds -- the question we dropped as not-ours-to-fix, reported here only
because the diagnostic is free."""
import numpy as np, xarray as xr, glob

P = "/scratch/db194/mom6/feb2026/channel_extra_sponge_slow_woc_p25"
CLAMP, SPONGE_LAT, SMAX = 15.0, -30.625, 0.01

def load(run, stream):
    f = sorted(glob.glob(f"{P}/{run}/output/{stream}_*.nc"))
    return xr.open_mfdataset(f, combine="by_coords", decode_times=False) if f else None

for run in ["tau_0.2_cb_1.0_cu_0.0_Aoff_diag", "tau_0.2_cb_1.0_cu_0.0_Boff_diag", "tau_0.2_cb_1.0_cu_0.0_Bon_diag"]:
    tag = {"Aoff_diag": "LOCAL_GRAD, clamp off", "Boff_diag": "STENCIL_GRAD, clamp off", "Bon_diag": "STENCIL_GRAD, clamp ON"}[run.split("_")[-2]+"_"+run.split("_")[-1]]
    snap = load(run, "ups_snap")
    if snap is None:
        print(f"{tag}: no snapshots yet"); continue
    lat = snap["yh"].values
    keep = lat < SPONGE_LAT
    U = np.abs(np.concatenate([snap["Upsilon_u"].isel(yh=keep).values.ravel(),
                               snap["Upsilon_v"].values.ravel()]))
    U = U[np.isfinite(U) & (U > 0)]
    q = np.percentile(U, [50, 90, 99, 99.9])
    print(f"\n=== {run.split('_')[-1]} ({tag.strip()}), instantaneous Upsilon, n={U.size}")
    print(f"  median {q[0]:.3f}  p90 {q[1]:.2f}  p99 {q[2]:.1f}  p99.9 {q[3]:.1f}  max {U.max():.1f} m2/s")
    print(f"  |Upsilon| > {CLAMP}: {100*np.mean(U > CLAMP):.2f}%   > 100: {100*np.mean(U > 100):.3f}%"
          f"   > 1e4: {100*np.mean(U > 1e4):.4f}%")

    dg = load(run, "ups_diag")
    if dg is None: continue
    for c in ("u", "v"):
        if f"s_hat_{c}" not in dg: break     # LOCAL_GRAD registers no s_hat/clamp diags
        m = dg[f"Upsilon_clamp_{c}"]
        if c == "u": m = m.isel(yh=keep)
        mv = m.values; mv = mv[np.isfinite(mv)]
        s = dg[f"s_hat_{c}"]
        if c == "u": s = s.isel(yh=keep)
        sv = np.abs(s.values); sv = sv[np.isfinite(sv)]
        print(f"  {c}-faces: clamp-mask time-mean {100*np.nanmean(mv):.2f}%   "
              f"s_hat median {np.median(sv):.5f} max {sv.max():.5f} (bound {3*SMAX:.2f}, "
              f"exceed {100*np.mean(sv > 3*SMAX + 1e-9):.3f}%)")
    # downstream mass limiter: realized vs unlimited streamfunction
    for c, uc in (("x", "GM_sfn_x"), ("y", "GM_sfn_y")):
        a = dg[uc].values; b = dg[f"GM_sfn_unlim_{c}"].values
        ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
        if ok.sum():
            print(f"  {c}: |realized/unlimited| median {np.median(np.abs(a[ok]/b[ok])):.4f}, "
                  f"limiter binds (ratio<0.99) on {100*np.mean(np.abs(a[ok]/b[ok]) < 0.99):.1f}%")

"""Pavel set MESO_UPSILON_CLAMP=15 m2/s from diagnosing CESM *online output*. Question (Dhruv):
what does the SAME diagnosis give on our TRAINING data — i.e. what Upsilon magnitudes do the
DIAGNOSED (true) subfilter fluxes of CM2.6 actually reach? If the true tail sits above 15, the
clamp truncates transports the training data itself contains; and the factor (Delta) dependence
shows whether any single dimensional value can be right across resolutions.

Construction mirrors the Fortran / train_rho_fluxes.ape_sink exactly: per-component
Upsilon_x = Fx / sqrt(rhox^2 + rhoz^2), rhoz = -(rho0/g) N^2, gradient floor 1e-10 (points below
the floor dropped, as the code zeroes them). Statistics over wet points, one test shard per factor;
full column and the flux band (z <= 500 m) separately."""
import numpy as np, xarray as xr

ROOT = "/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3"
RHO0, G, FLOOR = 1025.0, 9.8, 1.0e-10
QS = [50, 90, 99, 99.9, 99.99]

print(f"{'factor':>7}{'band':>10}{'n_wet':>10}{'med':>8}{'p90':>8}{'p99':>9}{'p99.9':>9}{'p99.99':>10}"
      f"{'max':>10}{'>10':>8}{'>15':>8}{'>35':>8}")
for fac in [4, 9, 12, 15]:
    ds = xr.open_dataset(f"{ROOT}/factor-{fac}/test-0.nc")
    z = ds["zl"].values
    Fx, Fy = ds["Fx"].values, ds["Fy"].values
    rx, ry = ds["rhox"].values, ds["rhoy"].values
    N2 = ds["N_buoyancy"].values ** 2
    ds.close()
    rhoz = -(RHO0 / G) * N2
    out = {}
    for comp, (F, r) in {"x": (Fx, rx), "y": (Fy, ry)}.items():
        mag = np.sqrt(r.astype("f8") ** 2 + rhoz.astype("f8") ** 2)
        U = np.abs(F.astype("f8")) / mag
        U[mag < FLOOR] = np.nan                      # the code zeroes these; drop from stats
        out[comp] = U
    U = np.concatenate([out["x"].ravel(), out["y"].ravel()])
    band = np.concatenate([np.broadcast_to(z[:, None, None], out["x"].shape).ravel()] * 2)
    for nm, sel in [("full", np.ones_like(U, bool)), ("z<=500m", band <= 500.0)]:
        u = U[sel]; u = u[np.isfinite(u)]
        q = np.percentile(u, QS)
        print(f"{fac:>7}{nm:>10}{u.size:>10}{q[0]:>8.3f}{q[1]:>8.2f}{q[2]:>9.1f}{q[3]:>9.1f}"
              f"{q[4]:>10.1f}{u.max():>10.1f}"
              f"{100*np.mean(u > 10):>7.2f}%{100*np.mean(u > 15):>7.2f}%{100*np.mean(u > 35):>7.2f}%")
    del Fx, Fy, rx, ry, N2, rhoz, out, U, band
print("\n(factor 4/9/12/15 = 0.4/0.9/1.2/1.5 deg. Clamp=15; GM ceiling kappa*s=10; "
      "our 1-deg GM-match ceiling ~35.)")

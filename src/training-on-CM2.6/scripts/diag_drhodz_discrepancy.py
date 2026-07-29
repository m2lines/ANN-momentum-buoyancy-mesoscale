"""Resolve the ~11x disagreement between the two CM2.6 d(rho)/dz estimates:
    raw   = np.gradient(rho, zl)                       median 5.1e-3 kg m-4
    N-route = (rho0/g) * N_buoyancy^2                  median 4.8e-4 kg m-4
Hypotheses tested here, in order:
  H1 dry cells: rho() returns wet*rho + (1-wet)*rho0, so a bottom-adjacent column sees a jump to
     1025 and np.gradient reports a spurious gradient there.
  H2 the N route interpolates N (not N^2) to centres, and squaring after averaging under-estimates.
  H3 a units/level-spacing problem in zl.
Prints a single deep open-ocean column both ways, then the medians restricted to points that are
strictly interior in the vertical, which isolates H1."""
import numpy as np, xarray as xr

ROOT = "/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-9"
RHO0, G = 1025.0, 9.8
d = xr.open_dataset(f"{ROOT}/test-0.nc"); p = xr.open_dataset(f"{ROOT}/param.nc")
rho = d["rho"].values.astype("f8"); zl = d["zl"].values.astype("f8")
N = d["N_buoyancy"].values.astype("f8")
wet = p["wet"].values > 0.5
print(f"zl: n={len(zl)} first={zl[:4]} last={zl[-3:]}  (dz range {np.diff(zl).min():.1f}-{np.diff(zl).max():.1f} m)")
print(f"rho: min={np.nanmin(rho):.2f} max={np.nanmax(rho):.2f} has_nan={np.isnan(rho).any()}")
print(f"rho at DRY points: median={np.median(rho[~wet]):.3f}  (H1 predicts exactly {RHO0})")
print(f"wet fraction of the 3-D grid: {100*wet.mean():.1f}%")

# --- H1: how deep is each wet column, and how many wet points sit next to the bottom?
kmax = np.where(wet.any(axis=0), wet.shape[0] - 1 - np.argmax(wet[::-1], axis=0), -1)
K = np.arange(wet.shape[0])[:, None, None]
interior = wet & (K < kmax[None] - 1) & (K > 0)          # strictly inside the water column
print(f"of wet points, {100*interior.sum()/wet.sum():.1f}% are vertically interior")

drdz_raw = np.gradient(rho, zl, axis=0)
drdz_N = (RHO0 / G) * N ** 2
for nm, m in [("all wet", wet), ("vertically interior", interior)]:
    a, b = np.abs(drdz_raw[m]), np.abs(drdz_N[m])
    print(f"  {nm:<22} raw median {np.median(a):.3e}   N-route median {np.median(b):.3e}"
          f"   ratio {np.median(a)/np.median(b):.1f}")

# --- a single deep open-ocean column, printed both ways
deep = np.argwhere(kmax > 40)
j, i = deep[len(deep) // 2]
k0 = int(kmax[j, i])
print(f"\ncolumn (j={j}, i={i}), bottom index {k0}, depth {zl[k0]:.0f} m")
print(f"{'k':>3}{'z [m]':>9}{'rho':>12}{'raw drdz':>12}{'N [1/s]':>11}{'N-route drdz':>14}{'wet':>5}")
for k in list(range(0, min(6, k0))) + list(range(max(0, k0 - 3), min(k0 + 3, len(zl)))):
    print(f"{k:>3}{zl[k]:>9.1f}{rho[k,j,i]:>12.4f}{drdz_raw[k,j,i]:>12.3e}"
          f"{N[k,j,i]:>11.3e}{drdz_N[k,j,i]:>14.3e}{int(wet[k,j,i]):>5}")

# --- H2: does squaring-after-interpolation explain a factor ~11? Rebuild N^2 at centres the
#     other way round (interpolate N^2 rather than N) using the raw profile as reference.
d2 = np.diff(rho, axis=0) / np.diff(zl)[:, None, None]        # at interfaces between centres
n2_if = (G / RHO0) * d2
n2_centre = np.full_like(rho, np.nan)
n2_centre[1:-1] = 0.5 * (n2_if[:-1] + n2_if[1:])
n_centre_sq = np.full_like(rho, np.nan)
n_centre_sq[1:-1] = (0.5 * (np.sqrt(np.maximum(n2_if[:-1], 0)) + np.sqrt(np.maximum(n2_if[1:], 0)))) ** 2
mm = interior & np.isfinite(n2_centre) & np.isfinite(n_centre_sq)
print(f"\nH2 check on the same field: interp(N^2) median {np.median(np.abs(n2_centre[mm])):.3e}"
      f"   [interp(N)]^2 median {np.median(np.abs(n_centre_sq[mm])):.3e}"
      f"   ratio {np.median(np.abs(n2_centre[mm]))/max(np.median(np.abs(n_centre_sq[mm])),1e-30):.2f}")
print(f"   stored N^2 median {np.median(N[mm]**2):.3e}  vs rebuilt-from-rho interp(N^2)"
      f" {np.median(np.abs(n2_centre[mm])):.3e}")
d.close(); p.close()

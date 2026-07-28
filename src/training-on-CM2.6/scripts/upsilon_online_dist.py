"""How often would MESO_UPSILON_CLAMP=15 actually fire, and how does that change with resolution?

Rather than waiting on MOM6 diagnostic runs (which only give 1/4deg), apply the ANN offline to a
STATE SNAPSHOT from each coarse channel run (1deg, 1/2deg, 1/4deg) and build the Upsilon
distribution the online code would see. The forward pass mirrors MOM_meso_sfn_ANN.F90 exactly:

  center gradients   drdx_c = 0.5*(drdx_u(i-1) + drdx_u(i))          [center_grad_rho]
  stencil norms      rho_grad_mag = ||[grad rho]_3x3||_2, vel_grad_mag = ||[grad u]_3x3||_2
  normalized inputs  each stencil divided by its own norm             [lines 388-393]
  re-dimensionalize  F = ANN_out * rho_grad_mag * vel_grad_mag * areaT * C_ANN   [line 411]
  conversion         Upsilon = Fx / sqrt(drdx^2 + drdz^2)             [LOCAL_GRAD, line 464-470]

Key scaling under test: F carries areaT, so Upsilon ~ Delta^2 |grad u| s -- a threshold fixed in
m2/s should bite far harder on coarse grids. Reported per resolution: distribution, fraction over
the clamp, and the GM-equivalent ceiling kappa_match*s_max from our own ladder."""
import numpy as np, xarray as xr, torch, sys
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
B = '/scratch/db194/mom6/feb2026'
RE, CB, CLAMP = 6.371e6, 1.0, 15.0
# (tag, nominal dlon=dlat in deg, GM-match kappa from Sec 4.1 -> GM ceiling kappa*0.01)
RES = [("1p0", 1.0, 3500.0), ("p5", 0.5, 2200.0), ("p25", 0.25, 1500.0)]
SPONGE_LAT = -30.625

def stencil(a, n=3):
    """(...,ny,nx) -> (...,ny,nx,n*n) 3x3 stencil, periodic in x, edge-replicated in y."""
    s = n // 2
    out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()
print(f"{'res':>5}{'n_wet':>10}{'med':>8}{'p90':>8}{'p99':>9}{'p99.9':>10}{'max':>11}"
      f"{'>15':>8}{'>GMceil':>9}{'GMceil':>8}")
rows = []
for tag, dd, kappa in RES:
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")                       # (z,y,x) center
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, z = ds["yh"].values, ds["z_l"].values
    ds.close()
    dy = np.deg2rad(dd) * RE
    dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    areaT = dx * dy
    # u is on xq (nx+1), v on yq (ny+1): face gradients -> center, matching center_grad_rho's 0.5 avg
    drdx_f = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx   # periodic x
    drdx_c = 0.5 * (drdx_f + np.roll(drdx_f, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    drdy_f = np.diff(rr, axis=-2) / dy
    drdy_c = 0.5 * (drdy_f[:, :-1] + drdy_f[:, 1:])
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx = (np.roll(uc, -1, axis=-1) - np.roll(uc, 1, axis=-1)) / (2 * dx)
    dvdx = (np.roll(vc, -1, axis=-1) - np.roll(vc, 1, axis=-1)) / (2 * dx)
    dudy = np.gradient(uc, axis=-2) / dy
    dvdy = np.gradient(vc, axis=-2) / dy
    sh_xx, sh_xy, vort = dudx - dvdy, dudy + dvdx, dvdx - dudy
    drdz = np.gradient(rho, z, axis=0)                            # (z,y,x)

    S = [stencil(f) for f in (drdx_c, drdy_c, sh_xx, sh_xy, vort)]
    rho_norm = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1))
    vel_norm = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
    ok = (rho_norm > 0) & (vel_norm > 0) & np.isfinite(rho_norm) & np.isfinite(vel_norm)
    x = np.concatenate([S[0] / rho_norm[..., None], S[1] / rho_norm[..., None],
                        S[2] / vel_norm[..., None], S[3] / vel_norm[..., None],
                        S[4] / vel_norm[..., None]], axis=-1)
    with torch.no_grad():
        out = ann(torch.from_numpy(x[ok])).numpy()                # (n,2) nondimensional
    Fx = np.full(rho_norm.shape, np.nan); Fy = np.full(rho_norm.shape, np.nan)
    pref = (rho_norm * vel_norm * np.broadcast_to(areaT, rho_norm.shape) * CB)[ok]
    Fx[ok] = -out[:, 0] * pref; Fy[ok] = -out[:, 1] * pref
    magx = np.sqrt(drdx_c ** 2 + drdz ** 2); magy = np.sqrt(drdy_c ** 2 + drdz ** 2)
    U = np.concatenate([np.abs(Fx) / magx, np.abs(Fy) / magy], axis=0).ravel()
    latm = np.concatenate([np.broadcast_to(lat[None, :, None], Fx.shape)] * 2, axis=0).ravel()
    U = U[np.isfinite(U) & (latm < SPONGE_LAT)]                   # drop the sponge band
    gm = kappa * 0.01
    q = np.percentile(U, [50, 90, 99, 99.9])
    print(f"{tag:>5}{U.size:>10}{q[0]:>8.3f}{q[1]:>8.2f}{q[2]:>9.2f}{q[3]:>10.1f}{U.max():>11.1f}"
          f"{100*np.mean(U > CLAMP):>7.2f}%{100*np.mean(U > gm):>8.2f}%{gm:>8.0f}")
    rows.append((tag, np.mean(U > CLAMP), np.median(U)))
print(f"\nclamp = {CLAMP} m2/s fixed; GMceil = (GM-match kappa)*s_max, which our Sec-4.1 ladder shows")
print("rises as the grid coarsens -- so a fixed clamp and the GM analogy diverge with resolution.")
print("median Upsilon ratio 1deg/(1/4deg):", f"{rows[0][2]/rows[2][2]:.1f}x  (Delta^2 would be 16x)")

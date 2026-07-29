"""FAITHFUL offline reconstruction of the quantity MESO_UPSILON_CLAMP is applied to, plus an
audit of how much each modelling choice matters. Prompted by Dhruv: "are you sure you computed
the same quantity the clamp is applied on?"

What MOM_meso_sfn_ANN.F90 actually clamps (LOCAL_GRAD, lines 462-492):
  * the ANN runs at CELL CENTRES: F_c = ANN * rho_grad_mag * vel_grad_mag * areaT * C_ANN
  * the flux is then interpolated to FACES:  Fx_u = 0.5*(F_c(i) + F_c(i+1))      [center2uv]
  * the divisor uses the NATIVE FACE gradients from calc_isoneutral_slopes:
        mag_grad = sqrt(drdx_u^2 + drdz_u^2)          <-- drdx_u, NOT the centred drdx_c
  * Upsilon_u = Fx_u / mag_grad   <-- THIS is what is compared against the clamp

My first pass used centre quantities throughout. Centre-averaging smooths the divisor away from
zero, so it biases the tail LOW; and a 2-cell centred velocity difference biases |grad u| low too.
Here both are fixed, and variants are reported side by side so the sensitivity is visible.

Remaining unavoidable gaps (stated, not hidden): rho is the z-remapped prog_z diagnostic on layer
midpoints rather than the native Z* interfaces; drdz comes from differencing that field rather than
from calc_isoneutral_slopes; no OBC/land masking beyond NaNs and the sponge. The definitive check
is the model's own meso_sfn_Upsilon_u diagnostic (run Bon_diag/Boff_diag)."""
import numpy as np, xarray as xr, torch, sys
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
B = '/scratch/db194/mom6/feb2026'
RE, CB, CLAMP, SPONGE_LAT = 6.371e6, 1.0, 15.0, -30.625
RES = [("1p0", 1.0), ("p5", 0.5), ("p25", 0.25)]

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()
print(f"{'res':>5} {'variant':<28}{'n':>9}{'med':>8}{'p90':>9}{'p99':>10}{'>15':>8}")
for tag, dd in RES:
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, z = ds["yh"].values, ds["z_l"].values; ds.close()
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]

    # ---- density gradients: FACE (native, as calc_isoneutral_slopes gives) and CENTRE (0.5 avg)
    drdx_u = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx     # at xq faces
    drdx_c = 0.5 * (drdx_u + np.roll(drdx_u, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    drdy_v = np.diff(rr, axis=-2) / dy                                                # at yq faces
    drdy_c = 0.5 * (drdy_v[:, :-1] + drdy_v[:, 1:])
    drdz = np.gradient(rho, z, axis=0)
    drdz_u = 0.5 * (drdz + np.roll(drdz, -1, axis=-1))                                # to faces
    drdz_v = 0.5 * (drdz[:, :-1] + drdz[:, 1:])

    # ---- velocity gradients: COMPACT from the face velocities (as MOM6), and the 2-cell version
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx_cp = np.diff(u, axis=-1) / dx                       # u on xq -> exact centre difference
    dvdy_cp = np.diff(v, axis=-2) / dy                       # v on yq -> exact centre difference
    dudy_cp = np.gradient(uc, axis=-2) / dy
    dvdx_cp = (np.roll(vc, -1, -1) - np.roll(vc, 1, -1)) / (2 * dx)
    dudx_w = (np.roll(uc, -1, -1) - np.roll(uc, 1, -1)) / (2 * dx)
    dvdy_w = np.gradient(vc, axis=-2) / dy

    def predict(dudx, dvdy, dudy, dvdx):
        S = [stencil(f) for f in (drdx_c, drdy_c, dudx - dvdy, dudy + dvdx, dvdx - dudy)]
        rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1))
        vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
        ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn)
        x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                            S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
        with torch.no_grad():
            out = ann(torch.from_numpy(x[ok])).numpy()
        Fx = np.full(rn.shape, np.nan); Fy = np.full(rn.shape, np.nan)
        pref = (rn * vn * np.broadcast_to(dx * dy, rn.shape) * CB)[ok]
        Fx[ok] = -out[:, 0] * pref; Fy[ok] = -out[:, 1] * pref
        return Fx, Fy

    Fx_cp, Fy_cp = predict(dudx_cp, dvdy_cp, dudy_cp, dvdx_cp)
    Fx_w, Fy_w = predict(dudx_w, dvdy_w, dudy_cp, dvdx_cp)

    latm = np.broadcast_to(lat[None, :, None], rho.shape)
    def report(name, Ux, Uy, maskx, masky):
        U = np.concatenate([Ux[maskx].ravel(), Uy[masky].ravel()])
        U = U[np.isfinite(U) & (U > 0)]
        q = np.percentile(U, [50, 90, 99])
        print(f"{tag:>5} {name:<28}{U.size:>9}{q[0]:>8.3f}{q[1]:>9.2f}{q[2]:>10.1f}"
              f"{100*np.mean(U > CLAMP):>7.1f}%")

    keep = latm < SPONGE_LAT
    # depth-resolved faithful fraction: the CM2.6 prototype shows clamping is a near-surface
    # phenomenon (24% at 5 m vs 1% at 288 m), so a pooled number hides the structure
    Fx_u0 = 0.5 * (Fx_cp + np.roll(Fx_cp, -1, axis=-1))
    Ux0 = np.abs(Fx_u0) / np.sqrt(drdx_u ** 2 + drdz_u ** 2)
    for zlo, zhi in [(0, 50), (50, 200), (200, 800), (800, 3000)]:
        m = (z >= zlo) & (z < zhi)
        if not m.any(): continue
        uu = Ux0[m][keep[m]]; uu = uu[np.isfinite(uu) & (uu > 0)]
        print(f"{tag:>5}   depth {zlo:>4}-{zhi:<4} m {'':>10}{np.median(uu):>8.3f}"
              f"{np.percentile(uu,90):>9.2f}{np.percentile(uu,99):>10.1f}{100*np.mean(uu>CLAMP):>7.1f}%")
    # (i) FAITHFUL: flux interpolated to faces, divisor = native face gradients
    Fx_u = 0.5 * (Fx_cp + np.roll(Fx_cp, -1, axis=-1))
    Fy_v = 0.5 * (Fy_cp[:, :-1] + Fy_cp[:, 1:])
    # drdy_v has ny+1 rows (both outer walls); interior faces are rows 1..ny-1, matching Fy_v
    drdy_int = drdy_v[:, 1:-1]
    report("faithful (face F, face grad)",
           np.abs(Fx_u) / np.sqrt(drdx_u ** 2 + drdz_u ** 2),
           np.abs(Fy_v) / np.sqrt(drdy_int ** 2 + drdz_v ** 2),
           keep, keep[:, :-1])
    # (ii) previous pass: everything at centres
    report("v1 (centre F, centre grad)",
           np.abs(Fx_cp) / np.sqrt(drdx_c ** 2 + drdz ** 2),
           np.abs(Fy_cp) / np.sqrt(drdy_c ** 2 + drdz ** 2), keep, keep)
    # (iii) isolate the velocity-gradient stencil choice
    report("v1 + 2-cell du/dx (old)",
           np.abs(Fx_w) / np.sqrt(drdx_c ** 2 + drdz ** 2),
           np.abs(Fy_w) / np.sqrt(drdy_c ** 2 + drdz ** 2), keep, keep)
print("\nSpread across variants = the systematic uncertainty of the offline reconstruction.")
print("Definitive number requires the model's own meso_sfn_Upsilon_u diagnostic.")

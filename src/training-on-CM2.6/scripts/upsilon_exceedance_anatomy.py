"""Dhruv's question: for the points where |Upsilon| exceeds the clamp, is it because the FLUX is
large or because the DIVISOR is small? And where are they -- surface, walls, topography?

Attribution: log|Upsilon| = log|F| - log|grad_3 rho|, so relative to the population median each
exceedance decomposes additively into a "flux is unusually large" part and a "divisor is unusually
small" part. Reporting both in decades makes the split unambiguous.

Geometry: the channel is lat -50..-27 (S wall at -50, sponge from -30.625), re-entrant in x, with a
meridional Gaussian ridge at lon 20 (crest 1500 m of 3000 m). Exceedances are binned by depth,
distance from the S wall, ridge proximity, and how close the point sits to the local bottom.
Faithful face-based construction (matches MOM_meso_sfn_ANN.F90: face flux / native face gradient)."""
import numpy as np, xarray as xr, torch, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
B = '/scratch/db194/mom6/feb2026'
RE, CB, CLAMP, SPONGE_LAT, WALL_S = 6.371e6, 1.0, 15.0, -30.625, -50.0

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()
for tag, dd in [("p25", 0.25), ("1p0", 1.0)]:
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, lon, z = ds["yh"].values, ds["xh"].values, ds["z_l"].values; ds.close()
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]

    drdx_u = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx
    drdx_c = 0.5 * (drdx_u + np.roll(drdx_u, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    drdy_c = 0.5 * (np.diff(rr, axis=-2)[:, :-1] + np.diff(rr, axis=-2)[:, 1:]) / dy
    drdz = np.gradient(rho, z, axis=0)
    drdz_u = 0.5 * (drdz + np.roll(drdz, -1, axis=-1))
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx = np.diff(u, axis=-1) / dx; dvdy = np.diff(v, axis=-2) / dy
    dudy = np.gradient(uc, axis=-2) / dy
    dvdx = (np.roll(vc, -1, -1) - np.roll(vc, 1, -1)) / (2 * dx)

    S = [stencil(f) for f in (drdx_c, drdy_c, dudx - dvdy, dudy + dvdx, dvdx - dudy)]
    rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1)); vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
    ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn)
    x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                        S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
    with torch.no_grad():
        out = ann(torch.from_numpy(x[ok])).numpy()
    Fc = np.full(rn.shape, np.nan)
    Fc[ok] = -out[:, 0] * (rn * vn * np.broadcast_to(dx * dy, rn.shape) * CB)[ok]
    Fu = 0.5 * (Fc + np.roll(Fc, -1, axis=-1))                  # face flux, as center2uv
    mag = np.sqrt(drdx_u ** 2 + drdz_u ** 2)                    # native face divisor
    U = np.abs(Fu) / mag

    Z = np.broadcast_to(z[:, None, None], U.shape)
    LAT = np.broadcast_to(lat[None, :, None], U.shape)
    LON = np.broadcast_to(lon[None, None, :], U.shape)
    wet = np.isfinite(U) & (U > 0) & (LAT < SPONGE_LAT)
    depth_col = np.where(np.isfinite(rho), Z, np.nan)
    bottom = np.nanmax(depth_col, axis=0)                       # deepest wet level per column
    HAB = np.broadcast_to(bottom[None], U.shape) - Z            # height above bottom

    lu, lf, lm = np.log10(U[wet]), np.log10(np.abs(Fu)[wet]), np.log10(mag[wet])
    hi = lu > np.log10(CLAMP)
    mf, mm = np.median(lf), np.median(lm)
    print(f"\n=== {tag}: {100*hi.mean():.1f}% of wet faces exceed {CLAMP} m2/s "
          f"(n={hi.sum()} of {hi.size}) ===")
    print(f"  attribution (decades vs population median):"
          f"  flux  {np.median(lf[hi]) - mf:+.2f}   divisor {-(np.median(lm[hi]) - mm):+.2f}"
          f"   -> {'DIVISOR' if -(np.median(lm[hi])-mm) > (np.median(lf[hi])-mf) else 'FLUX'}-dominated")
    print(f"  median |F|  all {10**mf:.3e}  exceed {10**np.median(lf[hi]):.3e} kg m-2 s-1")
    print(f"  median |grad3 rho| all {10**mm:.3e}  exceed {10**np.median(lm[hi]):.3e} kg m-4")
    # what fraction of exceedances are explained by a divisor below the 10th pct of the population?
    p10m, p90f = np.percentile(lm, 10), np.percentile(lf, 90)
    print(f"  of exceedances: {100*np.mean(lm[hi] < p10m):.0f}% have divisor in the population's "
          f"lowest decile; {100*np.mean(lf[hi] > p90f):.0f}% have flux in the highest decile")

    print("  WHERE (share of exceedances vs share of wet points):")
    Zw, HABw, LATw, LONw = Z[wet], HAB[wet], LAT[wet], LON[wet]
    for nm, sel in [("surface  z<50 m", Zw < 50), ("z 50-200 m", (Zw >= 50) & (Zw < 200)),
                    ("z 200-800 m", (Zw >= 200) & (Zw < 800)), ("z>800 m", Zw >= 800),
                    ("within 200 m of bottom", HABw < 200),
                    ("S-wall band (<1.5 deg)", LATw < WALL_S + 1.5),
                    ("ridge strip |lon-20|<3", np.abs(LONw - 20) < 3)]:
        print(f"    {nm:<26} {100*np.mean(sel[hi]):>5.1f}%   (wet {100*np.mean(sel):>5.1f}%)"
              f"   enrichment x{np.mean(sel[hi])/max(np.mean(sel),1e-9):.2f}")
    if tag == "p25":
        frac = np.where(wet.any(axis=0), (U > CLAMP).sum(axis=0) / np.maximum(wet.sum(axis=0), 1), np.nan)
        fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
        pc = ax[0].pcolormesh(lon, lat, 100 * frac, cmap="magma_r", shading="auto", vmin=0, vmax=60)
        plt.colorbar(pc, ax=ax[0], label="% of column clamped")
        ax[0].contour(lon, lat, bottom, levels=[1750, 2250, 2750], colors="c", linewidths=0.7)
        ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
        ax[0].set_title("(a) where the clamp engages (1/4$^\\circ$)", loc="left")
        prof = [100 * np.mean((U[k][wet[k]] > CLAMP)) if wet[k].any() else np.nan for k in range(len(z))]
        ax[1].plot(prof, z, "-o", ms=3, color="#08306b")
        ax[1].invert_yaxis(); ax[1].set_xlabel("% of faces clamped"); ax[1].set_ylabel("depth [m]")
        ax[1].set_title("(b) vertical structure", loc="left")
        ax[1].spines[["top", "right"]].set_visible(False); ax[1].grid(alpha=0.3)
        p = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/fig_upsilon_where.png"
        fig.savefig(p, dpi=150); print(f"  wrote {p}")

"""Figure for the Upsilon-clamp argument: where does MESO_UPSILON_CLAMP=15 sit in the distribution
of transports the scheme actually produces, and how much does it clip?

(a) Distribution of |Upsilon| (log axis) for the ANN applied to channel states at three resolutions,
    plus the DIAGNOSED (true) sub-grid transport of CM2.6 -- the same quantity Pavel diagnosed from
    CESM output when choosing 15. The clamp is drawn as a line with the clipped region shaded.
(b) Exceedance P(|Upsilon| > T) vs threshold T: reads off, for ANY candidate clamp value, the
    fraction of points it would engage. Markers show the fraction clipped at 15; open markers show
    the fraction clipped at the resolution-appropriate GM ceiling kappa_match * s_max.

The point of the pairing: the bulk of the distribution sits orders of magnitude below 15 and the
tail runs orders of magnitude above it, with no gap between -- so no threshold separates physical
transports from the ill-posed tail, and 15 lands inside the physical bulk. The truth curve shows the
tail is not an artifact of the network.
"""
import numpy as np, xarray as xr, torch, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
B = '/scratch/db194/mom6/feb2026'
RE, CB, CLAMP, SPONGE_LAT = 6.371e6, 1.0, 15.0, -30.625
RES = [("1p0", 1.0, 3500.0, "1$^\\circ$", "#9ecae1"),
       ("p5", 0.5, 2200.0, "1/2$^\\circ$", "#4292c6"),
       ("p25", 0.25, 1500.0, "1/4$^\\circ$", "#08306b")]
CM26_FACTOR, RHO0_T, G_T, FLOOR = 9, 1025.0, 9.8, 1.0e-10

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()
series = []
for tag, dd, kappa, lab, col in RES:
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, z = ds["yh"].values, ds["z_l"].values; ds.close()
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    drdx_f = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx
    drdx_c = 0.5 * (drdx_f + np.roll(drdx_f, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    drdy_f = np.diff(rr, axis=-2) / dy
    drdy_c = 0.5 * (drdy_f[:, :-1] + drdy_f[:, 1:])
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx = (np.roll(uc, -1, -1) - np.roll(uc, 1, -1)) / (2 * dx)
    dvdx = (np.roll(vc, -1, -1) - np.roll(vc, 1, -1)) / (2 * dx)
    dudy = np.gradient(uc, axis=-2) / dy; dvdy = np.gradient(vc, axis=-2) / dy
    S = [stencil(f) for f in (drdx_c, drdy_c, dudx - dvdy, dudy + dvdx, dvdx - dudy)]
    rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1)); vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
    ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn)
    x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                        S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
    with torch.no_grad():
        out = ann(torch.from_numpy(x[ok])).numpy()
    Fx = np.full(rn.shape, np.nan); Fy = np.full(rn.shape, np.nan)
    pref = (rn * vn * np.broadcast_to(dx * dy, rn.shape) * CB)[ok]
    Fx[ok] = -out[:, 0] * pref; Fy[ok] = -out[:, 1] * pref
    drdz = np.gradient(rho, z, axis=0)
    U = np.concatenate([np.abs(Fx) / np.sqrt(drdx_c ** 2 + drdz ** 2),
                        np.abs(Fy) / np.sqrt(drdy_c ** 2 + drdz ** 2)], axis=0).ravel()
    latm = np.concatenate([np.broadcast_to(lat[None, :, None], Fx.shape)] * 2, axis=0).ravel()
    U = U[np.isfinite(U) & (latm < SPONGE_LAT) & (U > 0)]
    series.append((lab, col, U, kappa * 0.01))
    print(f"channel {lab}: n={U.size} med={np.median(U):.3f} >15={100*np.mean(U>CLAMP):.1f}%", flush=True)

# CM2.6 diagnosed (true) sub-grid transport -- the same quantity, from the training data
d = xr.open_dataset(f"/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-{CM26_FACTOR}/test-0.nc")
rhoz = -(RHO0_T / G_T) * d["N_buoyancy"].values.astype("f8") ** 2
Ut = []
for Fk, rk in (("Fx", "rhox"), ("Fy", "rhoy")):
    mag = np.sqrt(d[rk].values.astype("f8") ** 2 + rhoz ** 2)
    t = np.abs(d[Fk].values.astype("f8")) / mag
    t[mag < FLOOR] = np.nan
    Ut.append(t.ravel())
d.close()
Ut = np.concatenate(Ut); Ut = Ut[np.isfinite(Ut) & (Ut > 0)]
print(f"CM2.6 diagnosed: n={Ut.size} med={np.median(Ut):.3f} >15={100*np.mean(Ut>CLAMP):.1f}%", flush=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
lb = np.linspace(-5, 6, 120)          # density per decade: histogram log10|Upsilon|

for lab, col, U, gm in series:
    ax[0].hist(np.log10(U), bins=lb, density=True, histtype="step", lw=1.6, color=col,
               label=f"ANN, {lab}")
ax[0].hist(np.log10(Ut), bins=lb, density=True, histtype="step", lw=1.6, color="k", ls="--",
           label="CM2.6 diagnosed (0.9$^\\circ$)")
ax[0].axvspan(np.log10(CLAMP), lb[-1], color="#d73027", alpha=0.08, lw=0)
ax[0].axvline(np.log10(CLAMP), color="#d73027", lw=1.4)
ax[0].annotate("clamp = 15", xy=(np.log10(CLAMP), 0.98), xycoords=("data", "axes fraction"),
               xytext=(-4, 0), textcoords="offset points", rotation=90, va="top", ha="right",
               color="#d73027", fontsize=8)
ax[0].set_xlim(-5, 6)
ax[0].set_xticks(np.arange(-4, 7, 2))
ax[0].set_xticklabels([f"$10^{{{k}}}$" for k in np.arange(-4, 7, 2)])
ax[0].set_xlabel("$|\\Upsilon|$ [m$^2$ s$^{-1}$]")
ax[0].set_ylabel("density per decade"); ax[0].set_title("(a) Distribution of transports", loc="left")
ax[0].legend(frameon=False, loc="upper left")
ax[0].spines[["top", "right"]].set_visible(False)

T = np.logspace(-2, 5, 220)
for lab, col, U, gm in series:
    ex = (U[None, :] > T[:, None]).mean(axis=1)
    ax[1].plot(T, 100 * ex, color=col, lw=1.8, label=f"ANN, {lab}")
    ax[1].plot([CLAMP], [100 * np.mean(U > CLAMP)], marker="o", ms=6, color=col, zorder=5)
    ax[1].plot([gm], [100 * np.mean(U > gm)], marker="o", ms=6, mfc="white", mec=col, mew=1.4, zorder=5)
ax[1].plot(T, 100 * (Ut[None, :] > T[:, None]).mean(axis=1), color="k", ls="--", lw=1.6,
           label="CM2.6 diagnosed (0.9$^\\circ$)")
ax[1].axvline(CLAMP, color="#d73027", lw=1.4)
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("clamp threshold $T$ [m$^2$ s$^{-1}$]")
ax[1].set_ylabel("points clamped [%]")
ax[1].set_title("(b) Fraction clipped at threshold $T$", loc="left")
ax[1].set_ylim(0.01, 100)
ax[1].legend(frameon=False, loc="lower left")
ax[1].spines[["top", "right"]].set_visible(False)
ax[1].annotate("filled: at the shipped 15\nopen: at GM ceiling $\\kappa_\\mathrm{match}s_\\mathrm{max}$",
               xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=7.5)

png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/fig_upsilon_clamp.png"
fig.savefig(png, dpi=150); fig.savefig(png.replace(".png", ".pdf"))
print("wrote", png)

"""SI figure for the flux-to-streamfunction conversion: where a fixed cap on Upsilon sits relative
to the transports the closure actually produces, in two very different stratification regimes.

(a) CM2.6, all four filter scales -- the regime the network was trained on and the one a realistic
    global model resembles. Unimodal; a cap at 15 m2/s clips the outer flank.
(b) the idealized channel, all three coarse resolutions -- bimodal, with a second lobe near
    1e2-1e3 m2/s produced where the LOCAL 3-D density gradient collapses in weakly stratified
    abyssal water. A cap at 15 falls in the saddle between the two lobes.
(c) exceedance vs threshold for one representative case from each, which reads off the clipped
    fraction for any candidate cap.

Conventions (both settled by earlier diagnostics, see the docstrings of
diag_drhodz_discrepancy.py and upsilon_online_faithful.py):
  * CM2.6 d rho/dz from N_buoyancy -- the dataset's `rho` is pressure-dependent, so its raw
    vertical gradient is dominated by adiabatic compression, not stratification.
  * channel Upsilon built at u FACES from a face-interpolated flux over the NATIVE face gradient,
    matching what MOM_meso_sfn_ANN.F90 actually compares against the clamp; the channel's linear
    equation of state has no compressibility so a raw d rho/dz is legitimate there.
Writes the PDF straight into the paper repo."""
import numpy as np, xarray as xr, torch, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
CM26 = "/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3"
B = "/scratch/db194/mom6/feb2026"
OUT = "/home/db194/mesoscale_b_ml_parameterization/figures"
CLAMP, RHO0, G, CB, RE, SPONGE_LAT = 15.0, 1025.0, 9.8, 1.0, 6.371e6, -30.625
BLU = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
SCALES = [(4, "0.4$^\\circ$"), (9, "0.9$^\\circ$"), (12, "1.2$^\\circ$"), (15, "1.5$^\\circ$")]
CHAN = [("1p0", 1.0, "1$^\\circ$"), ("p5", 0.5, "1/2$^\\circ$"), ("p25", 0.25, "1/4$^\\circ$")]

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()

def run_ann(S, rn, vn, ok):
    x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                        S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
    with torch.no_grad():
        return ann(torch.from_numpy(x[ok])).numpy()

cm26 = {}
for (fac, lab), col in zip(SCALES, BLU):
    d = xr.open_dataset(f"{CM26}/factor-{fac}/test-0.nc"); p = xr.open_dataset(f"{CM26}/factor-{fac}/param.nc")
    g = lambda k: d[k].values.astype("f8")
    wet = p["wet"].values > 0.5
    areaT = (p["dxT"].values * p["dyT"].values).astype("f8")
    rhox, rhoy = g("rhox"), g("rhoy")
    magx = np.sqrt(rhox ** 2 + ((RHO0 / G) * g("N_buoyancy") ** 2) ** 2)
    sh_xx, sh_xy, vort = g("sh_xx"), g("sh_xy_h"), g("rel_vort_h")
    acc = []
    for k in range(rhox.shape[0]):
        if not wet[k].any(): continue
        S = [stencil(f[k][None])[0] for f in (rhox, rhoy, sh_xx, sh_xy, vort)]
        rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1)); vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
        ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn) & wet[k]
        if not ok.any(): continue
        o = run_ann(S, rn, vn, ok)
        acc.append(np.abs(o[:, 0]) * (rn * vn * areaT * CB)[ok] / magx[k][ok])
    U = np.concatenate(acc)[::3]; U = U[np.isfinite(U) & (U > 0)]
    cm26[lab] = (col, U)
    print(f"CM2.6 {lab}: med {np.median(U):.3f}  p99 {np.percentile(U,99):.1f}  >15 {100*np.mean(U>CLAMP):.2f}%", flush=True)
    d.close(); p.close()

chan = {}
for (tag, dd, lab), col in zip(CHAN, BLU[1:]):
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, z = ds["yh"].values, ds["z_l"].values; ds.close()
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    drdx_u = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx
    rhox = 0.5 * (drdx_u + np.roll(drdx_u, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    rhoy = 0.5 * (np.diff(rr, axis=-2)[:, :-1] + np.diff(rr, axis=-2)[:, 1:]) / dy
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx = np.diff(u, axis=-1) / dx; dvdy = np.diff(v, axis=-2) / dy
    dudy = np.gradient(uc, axis=-2) / dy
    dvdx = (np.roll(vc, -1, -1) - np.roll(vc, 1, -1)) / (2 * dx)
    S = [stencil(f) for f in (rhox, rhoy, dudx - dvdy, dudy + dvdx, dvdx - dudy)]
    rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1)); vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
    ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn)
    o = run_ann(S, rn, vn, ok)
    Fc = np.full(rn.shape, np.nan)
    Fc[ok] = -o[:, 0] * (rn * vn * np.broadcast_to(dx * dy, rn.shape) * CB)[ok]
    Fu = 0.5 * (Fc + np.roll(Fc, -1, axis=-1))                      # face flux, as center2uv
    drdz = np.gradient(rho, z, axis=0)
    drdz_u = 0.5 * (drdz + np.roll(drdz, -1, axis=-1))
    U = (np.abs(Fu) / np.sqrt(drdx_u ** 2 + drdz_u ** 2))           # native face divisor
    keep = np.broadcast_to(lat[None, :, None], U.shape) < SPONGE_LAT
    U = U[keep]; U = U[np.isfinite(U) & (U > 0)]
    chan[lab] = (col, U)
    print(f"channel {lab}: med {np.median(U):.3f}  p99 {np.percentile(U,99):.1f}  >15 {100*np.mean(U>CLAMP):.2f}%", flush=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.3), constrained_layout=True)
lb = np.linspace(-6, 6, 130)
for a, dat, ttl, leg in [(ax[0], cm26, "(a) CM2.6, by filter scale", "filter scale"),
                         (ax[1], chan, "(b) channel, by resolution", "grid spacing")]:
    for lab, (col, U) in dat.items():
        a.hist(np.log10(U), bins=lb, density=True, histtype="step", lw=1.7, color=col, label=lab)
    CLR = "#d73027"
    a.axvspan(np.log10(CLAMP), lb[-1], color=CLR, alpha=0.08, lw=0)
    a.axvline(np.log10(CLAMP), color=CLR, lw=1.4)
    a.annotate("cap = 15", xy=(np.log10(CLAMP), 0.98), xycoords=("data", "axes fraction"),
               xytext=(-4, 0), textcoords="offset points", rotation=90, va="top", ha="right",
               color=CLR, fontsize=8)
    a.set_xlim(-6, 6); a.set_xticks(np.arange(-6, 7, 3))
    a.set_xticklabels([f"$10^{{{k}}}$" for k in np.arange(-6, 7, 3)])
    a.set_xlabel("$|\\Upsilon|$ [m$^2$ s$^{-1}$]"); a.set_ylabel("density per decade")
    a.set_title(ttl, loc="left"); a.legend(frameon=False, loc="upper left", title=leg, title_fontsize=8)
    a.spines[["top", "right"]].set_visible(False)

T = np.logspace(-3, 5, 240)
for lab, U, col, nm in [("0.9$^\\circ$", cm26["0.9$^\\circ$"][1], "#2171b5", "CM2.6 (0.9$^\\circ$)"),
                        ("1/4$^\\circ$", chan["1/4$^\\circ$"][1], "#238b45", "channel (1/4$^\\circ$)")]:
    ax[2].plot(T, 100 * (U[None, :] > T[:, None]).mean(axis=1), color=col, lw=1.9, label=nm)
    ax[2].plot([CLAMP], [100 * np.mean(U > CLAMP)], "o", ms=7, color=col, zorder=5)
ax[2].axvline(CLAMP, color="#d73027", lw=1.4)
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_ylim(1e-3, 100)
ax[2].set_xlabel("cap threshold $T$ [m$^2$ s$^{-1}$]"); ax[2].set_ylabel("points capped [%]")
ax[2].set_title("(c) fraction capped at threshold $T$", loc="left")
ax[2].legend(frameon=False, loc="lower left"); ax[2].spines[["top", "right"]].set_visible(False)

fig.savefig(f"{OUT}/si_upsilon_distribution.pdf", dpi=300)
fig.savefig("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/fig_si_upsilon.png", dpi=150)
print("wrote", f"{OUT}/si_upsilon_distribution.pdf")

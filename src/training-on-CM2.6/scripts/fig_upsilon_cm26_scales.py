"""CM2.6 only: how the transport Upsilon = F/|grad_3 rho| is distributed across ALL filter scales
(factors 4, 9, 12, 15 -> 0.4, 0.9, 1.2, 1.5 deg), for both the DIAGNOSED (true) sub-grid flux and
the ANN PREDICTION. Shows where a fixed clamp sits relative to the physics as the filter widens.

d rho/dz convention: the raw vertical gradient of the dataset's own `rho`, i.e. the SAME treatment
used for the channel, so the two figures are comparable. Note the pipeline's N_buoyancy route gives
a systematically SMALLER |d rho/dz| (and hence LARGER Upsilon) -- both are printed so the
sensitivity is on the record; the raw route used here is the conservative one."""
import numpy as np, xarray as xr, torch, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
ROOT = "/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3"
CLAMP, RHO0_T, G_T, CB = 15.0, 1025.0, 9.8, 1.0
SCALES = [(4, "0.4$^\\circ$", "#c6dbef"), (9, "0.9$^\\circ$", "#6baed6"),
          (12, "1.2$^\\circ$", "#2171b5"), (15, "1.5$^\\circ$", "#08306b")]

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()
res = {}
print(f"{'scale':>7}{'source':>12}{'med':>10}{'p90':>10}{'p99':>11}{'>15':>8}   (|drdz| median)")
for fac, lab, col in SCALES:
    d = xr.open_dataset(f"{ROOT}/factor-{fac}/test-0.nc")
    p = xr.open_dataset(f"{ROOT}/factor-{fac}/param.nc")
    g = lambda k: d[k].values.astype("f8")
    wet = p["wet"].values > 0.5
    areaT = (p["dxT"].values * p["dyT"].values).astype("f8")[None]
    rhox, rhoy = g("rhox"), g("rhoy")
    drdz_raw = np.gradient(g("rho"), d["zl"].values, axis=0)
    drdz_N = -(RHO0_T / G_T) * g("N_buoyancy") ** 2
    magx_raw = np.sqrt(rhox ** 2 + drdz_raw ** 2)
    magx_N = np.sqrt(rhox ** 2 + drdz_N ** 2)

    # Level-by-level so the 45-column stencil array never exceeds ~100 MB (factor-4 whole-field
    # would be ~10 GB); STRIDE thins the accumulated samples, which a distribution does not need.
    STRIDE = 3
    sh_xx, sh_xy, vort = g("sh_xx"), g("sh_xy_h"), g("rel_vort_h")
    Fx = g("Fx")
    accD, accP, accPN = [], [], []
    for k in range(rhox.shape[0]):
        wk = wet[k]
        if not wk.any(): continue
        S = [stencil(f[k][None])[0] for f in (rhox, rhoy, sh_xx, sh_xy, vort)]
        rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1)); vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
        ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn) & wk
        if not ok.any(): continue
        x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                            S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
        with torch.no_grad():
            o = ann(torch.from_numpy(x[ok])).numpy()
        pref = np.abs(o[:, 0]) * (rn * vn * np.broadcast_to(areaT[0], rn.shape) * CB)[ok]
        accP.append(pref / magx_raw[k][ok]); accPN.append(pref / magx_N[k][ok])
        accD.append((np.abs(Fx[k]) / magx_raw[k])[wk])
    fin = lambda L: (lambda a: a[np.isfinite(a) & (a > 0)])(np.concatenate(L)[::STRIDE])
    Ud, Up, UpN = fin(accD), fin(accP), fin(accPN)
    res[fac] = (lab, col, Ud, Up)
    for nm, a, dz in [("diagnosed", Ud, drdz_raw), ("ANN", Up, drdz_raw),
                      ("ANN (N_buoy drdz)", UpN, drdz_N)]:
        q = np.percentile(a, [50, 90, 99])
        print(f"{lab:>7}{nm:>12}{q[0]:>10.3f}{q[1]:>10.2f}{q[2]:>11.1f}{100*np.mean(a > CLAMP):>7.2f}%"
              f"   {np.median(np.abs(dz[wet])):.2e}")
    d.close(); p.close()

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
lb = np.linspace(-6, 5, 120)
for fac, (lab, col, Ud, Up) in res.items():
    ax[0].hist(np.log10(Up), bins=lb, density=True, histtype="step", lw=1.7, color=col, label=lab)
    ax[0].hist(np.log10(Ud), bins=lb, density=True, histtype="step", lw=1.1, ls=":", color=col)
ax[0].axvspan(np.log10(CLAMP), lb[-1], color="#d73027", alpha=0.08, lw=0)
ax[0].axvline(np.log10(CLAMP), color="#d73027", lw=1.4)
ax[0].annotate("clamp = 15", xy=(np.log10(CLAMP), 0.98), xycoords=("data", "axes fraction"),
               xytext=(-4, 0), textcoords="offset points", rotation=90, va="top", ha="right",
               color="#d73027", fontsize=8)
ax[0].set_xlim(-6, 5); ax[0].set_xticks(np.arange(-6, 6, 2))
ax[0].set_xticklabels([f"$10^{{{k}}}$" for k in np.arange(-6, 6, 2)])
ax[0].set_xlabel("$|\\Upsilon|$ [m$^2$ s$^{-1}$]"); ax[0].set_ylabel("density per decade")
ax[0].set_title("(a) CM2.6 transports by filter scale", loc="left")
ax[0].legend(frameon=False, loc="center left", title="solid: ANN\ndotted: diagnosed",
             title_fontsize=7.5)
ax[0].spines[["top", "right"]].set_visible(False)

T = np.logspace(-3, 5, 220)
for fac, (lab, col, Ud, Up) in res.items():
    ax[1].plot(T, 100 * (Up[None, :] > T[:, None]).mean(axis=1), color=col, lw=1.8, label=lab)
    ax[1].plot(T, 100 * (Ud[None, :] > T[:, None]).mean(axis=1), color=col, lw=1.1, ls=":")
    ax[1].plot([CLAMP], [100 * np.mean(Up > CLAMP)], "o", ms=6, color=col, zorder=5)
ax[1].axvline(CLAMP, color="#d73027", lw=1.4)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-3, 100)
ax[1].set_xlabel("clamp threshold $T$ [m$^2$ s$^{-1}$]"); ax[1].set_ylabel("points clamped [%]")
ax[1].set_title("(b) fraction clipped at threshold $T$", loc="left")
ax[1].legend(frameon=False, loc="lower left")
ax[1].spines[["top", "right"]].set_visible(False)

np.savez("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/upsilon_cm26_scales.npz",
         **{f"{f}_{k}": v for f, (lab, col, Ud, Up) in res.items() for k, v in (("diag", Ud), ("ann", Up))})
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/fig_upsilon_cm26_scales.png"
fig.savefig(png, dpi=150); print("wrote", png)

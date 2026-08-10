"""NW2 closures against the 1/32-degree truth -- the same framing as the channel's Sec 4.1 figure.

Perezhogin's filtered-and-coarsened truth (`R32/0.5-degree-coarsen-snapshots.nc`) sits on a
BIT-IDENTICAL grid to our R2 runs (120x280x15, same cell centres), so it differences directly with
no regridding.

    target   = truth - no-closure      <- what resolving the eddies actually does
    response = run   - no-closure      <- what each closure does instead

A closure that works reproduces the target: regression slope near 1 and high pattern correlation.

APE uses the CORRECTED functional (nw2_ape_check.py): reference -H0, MOM6's adiabatically-flattened
state, and the topography term removed --
    APE = sum_i 0.5 g drho_i [ (e_i - e_rest)^2 - max(e_bot - e_rest, 0)^2 ]
Truth interfaces are rebuilt from its `h` by cumulative sum, which is exact by definition.

TWO HONEST LIMITS, both from what is on disk: the truth file holds only 4 snapshots (days 2705-2795)
so its EKE is thinly sampled, and those days are not synchronous with our runs (days 30000-37200),
so this is a comparison against a climatological reference rather than a matched time. A background
job is coarsening R32/longmean*.nc (18 records, carries `e`, `uh/vh`, `KE`) to firm this up."""
import numpy as np, xarray as xr, glob, string
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, cmocean

B = "/scratch/db194/mom6/jul2026_nw2"
T = "/scratch/pp2681/mom6/Neverworld2/simulations/R32/0.5-degree-coarsen-snapshots.nc"
NWIN, RHO0 = 5, 1035.0
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"), ("MEKE", "MEKE")]

def last(r, s, n=NWIN): return sorted(glob.glob(f"{B}/{r}/output/{s}_*.nc"))[-n:]

d0 = xr.open_dataset(last("bare", "longmean")[-1], decode_times=False)
rho_l, lat, lon = d0["zl"].values, d0["yh"].values, d0["xh"].values; d0.close()
drho = np.diff(rho_l)
st = xr.open_dataset(f"{B}/bare/output/ocean.stats.nc", decode_times=False)
e_rest = -st["H0"].values[-200:].mean(axis=0); st.close()

from nw2_common import ape_map as _ape_corrected
def ape(e):
    return _ape_corrected(e, e_rest, drho)

def run_fields(r):
    e = np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values
                 for f in last(r, "longmean")], axis=0)
    eke = []
    for f in last(r, "snapshots"):
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values; d.close()
        uc = 0.5*(u[..., :-1]+u[..., 1:]); vc = 0.5*(v[:, :, :-1, :]+v[:, :, 1:, :])
        KE = 0.5*np.nanmean((uc**2+vc**2)*h, 0)
        um, vm, hm = np.nanmean(uc, 0), np.nanmean(vc, 0), np.nanmean(h, 0)
        eke.append(np.nansum(KE - 0.5*(um**2+vm**2)*hm, 0)/np.nansum(hm, 0))
    return ape(e), np.mean(eke, 0)

t = xr.open_dataset(T, decode_times=False)
h_t = t["h"].values                                   # (time, zl, yh, xh)
u_t, v_t = t["u"].values, t["v"].values
t.close()
hm_t = np.nanmean(h_t, 0)
e_t = np.concatenate([np.zeros((1,)+hm_t.shape[1:]), -np.cumsum(hm_t, axis=0)], axis=0)
KE_t = 0.5*np.nanmean((u_t**2 + v_t**2)*h_t, 0)
um_t, vm_t = np.nanmean(u_t, 0), np.nanmean(v_t, 0)
eke_t = np.nansum(KE_t - 0.5*(um_t**2+vm_t**2)*hm_t, 0)/np.nansum(hm_t, 0)
A_t = ape(e_t)

A_b, E_b = run_fields("bare")
tgt_a, tgt_e = A_t - A_b, eke_t - E_b
res = {r: tuple(np.subtract(run_fields(r), (A_b, E_b))) for r, _ in ORDER}

w = np.cos(np.deg2rad(lat))[:, None]
am = lambda x: float(np.nansum(x*w)/np.nansum(w*np.isfinite(x)))
def fit(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[m], y[m], 1)[0], np.corrcoef(x[m], y[m])[0, 1]
print(f"{'run':<10}{'dAPE slope':>12}{'r':>8}{'dEKE slope':>13}{'r':>8}{'dAPE mean':>12}{'target':>10}")
print(f"{'TARGET':<10}{1.0:>12.2f}{1.0:>8.2f}{1.0:>13.2f}{1.0:>8.2f}{am(tgt_a)/1e3:>12.1f}{'':>10}")
for r, lab in ORDER:
    da, de = res[r]
    sa, ra = fit(tgt_a.ravel(), da.ravel()); se, re_ = fit(tgt_e.ravel(), de.ravel())
    print(f"{r:<10}{sa:>12.2f}{ra:>8.2f}{se:>13.2f}{re_:>8.2f}{am(da)/1e3:>12.1f}{am(tgt_a)/1e3:>10.1f}")

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
n = len(ORDER)+1
fig, ax = plt.subplots(2, n, figsize=(2.35*n, 9.0), constrained_layout=True)
def panel(a, f, norm, cmap, first, lastrow):
    im = a.pcolormesh(lon, lat, f, norm=norm, cmap=cmap, shading="auto", rasterized=True)
    a.set_aspect("equal"); lats=[-60,-40,-20,0,20,40,60]
    a.set_yticks(lats); a.set_yticklabels([f"${abs(l)}^\\circ$S" if l<0 else (f"${l}^\\circ$N" if l>0 else "$0^\\circ$") for l in lats] if first else [""]*7)
    a.set_xticks([10,30,50]); a.set_xticklabels([f"${x}^\\circ$E" for x in [10,30,50]] if lastrow else ["","",""])
    return im
for i, (tg, sc, cmap, lab) in enumerate([(tgt_a, 1e-6, plt.cm.PuOr_r, "APE  [MJ m$^{-2}$]"),
                                         (tgt_e, 1e4, cmocean.cm.curl, "depth-avg eddy KE  [cm$^2$ s$^{-2}$]")]):
    allf = [tg*sc] + [res[r][i]*sc for r, _ in ORDER]
    lt = np.percentile(np.abs(np.concatenate([a.ravel() for a in allf])[np.isfinite(np.concatenate([a.ravel() for a in allf]))]), 60)
    vmax = np.percentile(np.abs(np.concatenate([a.ravel() for a in allf])[np.isfinite(np.concatenate([a.ravel() for a in allf]))]), 99.8)
    nm = mpl.colors.SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
    im = panel(ax[i, 0], tg*sc, nm, cmap, True, i == 1)
    for j, (r, _) in enumerate(ORDER, start=1):
        im = panel(ax[i, j], res[r][i]*sc, nm, cmap, False, i == 1)
    fig.colorbar(im, ax=ax[i, :], orientation="horizontal", extend="both", aspect=60,
                 pad=0.02).set_label(f"$\\Delta$ {lab}   (vs no closure)", fontsize=9)
ax[0, 0].set_title("TARGET: $\\overline{1/32^\\circ}$", loc="left")
for j, (r, lab) in enumerate(ORDER, start=1): ax[0, j].set_title(lab, loc="left")
for k, a in enumerate(ax.ravel()):
    a.text(0.04, 0.975, f"({string.ascii_lowercase[k]})", transform=a.transAxes, va="top",
           fontsize=10, fontweight="bold")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_vs_truth.png"
fig.savefig(png, dpi=150, bbox_inches="tight"); print("\nwrote", png)

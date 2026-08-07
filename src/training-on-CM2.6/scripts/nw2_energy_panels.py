"""NW2 energy response, in the panel style of Perezhogin et al.'s Figure 2.

Style borrowed deliberately from notebooks/Figure-2-and-S1.ipynb so the buoyancy paper's NW2 figure
reads as a companion to the momentum one:
  * SymLogNorm colour scaling (linear within +-linthresh, logarithmic beyond) -- essential here
    because the response spans the quiet subtropical interior and the energetic channel band across
    several decades, which a linear scale flattens to white;
  * equal aspect, so the basin has its true shape;
  * cmocean maps -- `curl` for eddy energy, `PuOr_r` for APE, `balance` for the mean flow;
  * one thin shared horizontal colourbar per row, extend='both';
  * geographic tick labels, only on the outer axes; panel letters.

Quantities as in nw2_ape_ke_maps.py: interface-displacement APE, and Perezhogin's thickness-weighted
depth-averaged MKE/EKE. Column 1 is the unparameterized control (absolute); the rest are responses.
ANN C=2 omitted (thin-layer transient, nw2_spike_diagnosis.py)."""
import numpy as np, xarray as xr, glob, string
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, cmocean

B = "/scratch/db194/mom6/jul2026_nw2"
G, NWIN, RHO0 = 9.8, 5, 1035.0
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"), ("MEKE", "MEKE")]

def last(run, s, n=NWIN): return sorted(glob.glob(f"{B}/{run}/output/{s}_*.nc"))[-n:]

def energy(run):
    mke, eke = [], []
    for f in last(run, "snapshots"):
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values; d.close()
        uc = 0.5*(u[..., :-1] + u[..., 1:]); vc = 0.5*(v[:, :, :-1, :] + v[:, :, 1:, :])
        KE = 0.5*np.nanmean((uc**2 + vc**2)*h, axis=0)
        um, vm, hm = np.nanmean(uc, 0), np.nanmean(vc, 0), np.nanmean(h, 0)
        MK = 0.5*(um**2 + vm**2)*hm; H = np.nansum(hm, axis=0)
        mke.append(np.nansum(MK, axis=0)/H); eke.append(np.nansum(KE-MK, axis=0)/H)
    return np.mean(mke, 0), np.mean(eke, 0)

def emean(run):
    return np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values
                    for f in last(run, "longmean")], axis=0)

d0 = xr.open_dataset(last("bare", "longmean")[-1], decode_times=False)
rho_l, lat, lon = d0["zl"].values, d0["yh"].values, d0["xh"].values; d0.close()
from nw2_common import load_e_rest, ape_map as _ape_corrected   # corrected APE (2026-08-07)
drho = np.diff(rho_l); e0 = emean("bare"); e_rest = load_e_rest(B)
ape = lambda e: _ape_corrected(e, e_rest, drho)
A0 = ape(e0); M0, E0 = energy("bare")
res = {r: (ape(emean(r))-A0, energy(r)[0]-M0, energy(r)[1]-E0) for r, _ in ORDER}

def panel(ax, fld, norm, cmap, first_col, last_row):
    im = ax.pcolormesh(lon, lat, fld, norm=norm, cmap=cmap, shading="auto", rasterized=True)
    ax.set_aspect("equal")
    lats = [-60, -40, -20, 0, 20, 40, 60]
    ax.set_yticks(lats)
    ax.set_yticklabels([f"${abs(l)}^\\circ$S" if l < 0 else (f"${l}^\\circ$N" if l > 0 else "$0^\\circ$")
                        for l in lats] if first_col else [""]*len(lats))
    ax.set_xticks([10, 30, 50])
    ax.set_xticklabels([f"${x}^\\circ$E" for x in [10, 30, 50]] if last_row else ["", "", ""])
    return im

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
n = len(ORDER) + 1
fig, ax = plt.subplots(3, n, figsize=(2.35*n, 12.6), constrained_layout=True)

# linthresh set FROM THE DATA (median |response| pooled over closures) so the bulk of the domain
# sits in the near-linear white zone and only the genuine signal is log-stretched; a linthresh
# below the noise floor turns sampling noise into visual speckle.
def scales(i, sc):
    a = np.abs(np.concatenate([ (res[r][i]*sc).ravel() for r, _ in ORDER ]))
    a = a[np.isfinite(a)]
    return np.percentile(a, 60), np.percentile(a, 99.8)
ROWS = []
for i, ctl, cctl, cdif, sc, lab in [
        (0, A0/1e6, cmocean.cm.matter, plt.cm.PuOr_r, 1e-6, "APE  [MJ m$^{-2}$]"),
        (1, M0*1e4, cmocean.cm.speed, cmocean.cm.balance, 1e4, "depth-avg mean KE  [cm$^2$ s$^{-2}$]"),
        (2, E0*1e4, cmocean.cm.speed, cmocean.cm.curl, 1e4, "depth-avg eddy KE  [cm$^2$ s$^{-2}$]")]:
    lt, vm = scales(i, sc)
    print(f"row {i}: linthresh {lt:.3g}, vmax {vm:.3g}")
    ROWS.append((i, ctl, cctl, cdif, lt, vm, lab))
for i, ctl, cctl, cdif, lt, vm, lab in ROWS:
    cn = mpl.colors.LogNorm(vmin=max(np.nanpercentile(ctl, 5), 1e-3), vmax=np.nanpercentile(ctl, 99.5))
    im_c = panel(ax[i, 0], np.where(ctl > 0, ctl, np.nan), cn, cctl, True, i == 2)
    fig.colorbar(im_c, ax=ax[i, 0], orientation="horizontal", extend="both",
                 aspect=18, pad=0.02).set_label(f"control {lab}", fontsize=8)
    dn = mpl.colors.SymLogNorm(linthresh=lt, vmin=-vm, vmax=vm, base=10)
    for j, (r, _) in enumerate(ORDER, start=1):
        f = res[r][i] * (1e-6 if i == 0 else 1e4)
        im_d = panel(ax[i, j], f, dn, cdif, False, i == 2)
    fig.colorbar(im_d, ax=ax[i, 1:], orientation="horizontal", extend="both",
                 aspect=60, pad=0.02).set_label(f"$\\Delta$ {lab}   (response to closure)", fontsize=9)

ax[0, 0].set_title("no closure", loc="left")
for j, (r, lab) in enumerate(ORDER, start=1):
    ax[0, j].set_title(lab, loc="left")
for k, a in enumerate(ax.ravel()):
    a.text(0.04, 0.975, f"({string.ascii_lowercase[k]})", transform=a.transAxes,
           va="top", fontsize=10, fontweight="bold")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_energy_panels.png"
fig.savefig(png, dpi=150, bbox_inches="tight"); print("wrote", png)

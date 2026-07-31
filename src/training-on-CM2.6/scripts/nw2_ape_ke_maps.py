"""NW2 response maps: APE, and depth-averaged kinetic energy split into mean and eddy parts.

Replaces the barotropic-streamfunction view with the energy metrics themselves. The KE convention
follows Perezhogin's NW2/OM4 analysis (notebooks/Figure-3.ipynb `kinetic_energy`), i.e.
thickness-weighted and time-averaged, with the mean part removed to isolate the eddies:

    KE  = 0.5 * <u^2 h>_t          MKE = 0.5 * <u>_t^2 <h>_t          EKE = KE - MKE

summed over layers and divided by the total depth, so the maps are depth-averaged [m2 s-2]. This
is a stricter EKE than a plain snapshot variance because the thickness weighting matters in an
isopycnal model where layers breathe.

APE is the interface-displacement form used throughout (see nw2_ape_eke_maps.py). Everything is
differenced against `bare`, a true zero-closure reference in NW2. ANN C=2 omitted -- thin-layer
transient (nw2_spike_diagnosis.py)."""
import numpy as np, xarray as xr, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/jul2026_nw2"
G, NWIN = 9.8, 5
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"), ("MEKE", "MEKE")]

def last(run, stream, n=NWIN):
    return sorted(glob.glob(f"{B}/{run}/output/{stream}_*.nc"))[-n:]

def energy(run):
    """depth-averaged MKE and EKE [m2/s2], Perezhogin's thickness-weighted convention."""
    mke, eke = [], []
    for f in last(run, "snapshots"):
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values
        d.close()
        uc = 0.5 * (u[..., :-1] + u[..., 1:])                  # xq -> xh
        vc = 0.5 * (v[:, :, :-1, :] + v[:, :, 1:, :])          # yq -> yh
        KE = 0.5 * np.nanmean((uc**2 + vc**2) * h, axis=0)     # <u^2 h>_t, per layer [m3/s2]
        um, vm, hm = np.nanmean(uc, 0), np.nanmean(vc, 0), np.nanmean(h, 0)
        MK = 0.5 * (um**2 + vm**2) * hm
        H = np.nansum(hm, axis=0)
        mke.append(np.nansum(MK, axis=0) / H)
        eke.append(np.nansum(KE - MK, axis=0) / H)
    return np.mean(mke, axis=0), np.mean(eke, axis=0)

def ape(run):
    fs = last(run, "longmean")
    e = np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values for f in fs], axis=0)
    return e

d0 = xr.open_dataset(last("bare", "longmean")[-1], decode_times=False)
rho_l, lat, lon = d0["zl"].values, d0["yh"].values, d0["xh"].values
d0.close()
drho = np.diff(rho_l)
e_bare = ape("bare")
R = np.nanmean(e_bare[1:-1], axis=(1, 2))
ape_map = lambda e: np.nansum(0.5 * G * drho[:, None, None] * (e[1:-1] - R[:, None, None])**2, axis=0)

A0 = ape_map(e_bare); M0, E0 = energy("bare")
res = {r: (ape_map(ape(r)) - A0, energy(r)[0] - M0, energy(r)[1] - E0) for r, _ in ORDER}

w = np.cos(np.deg2rad(lat))[:, None]
am = lambda x: float(np.nansum(x * w) / np.nansum(w * np.isfinite(x)))
print(f"{'run':<10}{'dAPE [kJ/m2]':>14}{'dMKE [cm2/s2]':>15}{'dEKE [cm2/s2]':>15}{'EKE %bare':>11}")
print(f"{'bare':<10}{'-':>14}{'-':>15}{'-':>15}{f'{1e4*am(E0):.1f} abs':>11}")
for r, lab in ORDER:
    da, dm, de = res[r]
    print(f"{r:<10}{am(da)/1e3:>14.1f}{1e4*am(dm):>15.2f}{1e4*am(de):>15.2f}"
          f"{100*(am(E0)+am(de))/am(E0):>10.0f}%")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
n = len(ORDER)
fig, ax = plt.subplots(3, n, figsize=(3.0*n, 9.2), sharex=True, sharey=True, constrained_layout=True)
va = np.nanpercentile(np.abs(np.stack([res[r][0] for r,_ in ORDER])), 99) / 1e6
vm = np.nanpercentile(np.abs(np.stack([res[r][1] for r,_ in ORDER])), 99) * 1e4
ve = np.nanpercentile(np.abs(np.stack([res[r][2] for r,_ in ORDER])), 99) * 1e4
for j, (r, lab) in enumerate(ORDER):
    da, dm, de = res[r]
    q0 = ax[0, j].pcolormesh(lon, lat, da/1e6, cmap="PuOr_r", vmin=-va, vmax=va, shading="auto", rasterized=True)
    q1 = ax[1, j].pcolormesh(lon, lat, dm*1e4, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    q2 = ax[2, j].pcolormesh(lon, lat, de*1e4, cmap="RdBu_r", vmin=-ve, vmax=ve, shading="auto", rasterized=True)
    ax[0, j].set_title(lab, loc="left"); ax[2, j].set_xlabel("longitude")
fig.colorbar(q0, ax=ax[0, :], shrink=0.85, label="$\\Delta$APE [MJ m$^{-2}$]")
fig.colorbar(q1, ax=ax[1, :], shrink=0.85, label="$\\Delta$MKE [cm$^2$ s$^{-2}$]")
fig.colorbar(q2, ax=ax[2, :], shrink=0.85, label="$\\Delta$EKE [cm$^2$ s$^{-2}$]")
for a, t in zip(ax[:, 0], ["available potential energy", "depth-avg MEAN KE", "depth-avg EDDY KE"]):
    a.set_ylabel(f"{t}\nlatitude")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_ape_ke_maps.png"
fig.savefig(png, dpi=140); print("\nwrote", png)

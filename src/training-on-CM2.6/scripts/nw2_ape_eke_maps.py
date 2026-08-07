"""NW2 analogue of the channel's dAPE / dEKE map comparison (Sec 4.1 Fig channel_ape_eke).

NW2 is ISOPYCNAL, so the interface-displacement APE the channel had to build in rho2 diagnostic
space is native here: the model's own `e` are the isopycnal interfaces and the layer densities are
fixed targets (zl coordinate), so

    APE(x,y) = sum_i 0.5 * g * drho_i * (e_i - R_i)^2      [J/m2]

with drho_i the density jump across interface i and R_i a common reference profile. There being no
truth comparison by design (Dhruv 2026-07-29: measure the RESPONSE to the parameterization, not the
match to a high-res truth), the reference is the horizontal mean of the unparameterized run's
time-mean interfaces -- a stated convention, since APE is quadratic and the reference does not
cancel in differences.

dEKE is the column-integrated resolved eddy energy from snapshot variance,
rho0 * sum_k 0.5 * h_k * (var_t u + var_t v), exactly as in the channel.

Everything is differenced against `bare`, which in NW2 is a TRUE zero-closure reference (MOM6
defaults THICKNESSDIFFUSE=False, KHTH=0 and the NW2 config never sets them)."""
import numpy as np, xarray as xr, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/jul2026_nw2"
RHO0, G, NWIN = 1035.0, 9.8, 5
# paired for comparison: matched-response pair first, then the stronger pair, then MEKE
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"),
         ("MEKE", "MEKE"), ("ANN_c2p0", "ANN $C$=2")]

def last(run, stream, n=NWIN):
    return sorted(glob.glob(f"{B}/{run}/output/{stream}_*.nc"))[-n:]

def emean(run):
    fs = last(run, "longmean")
    return np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values for f in fs], axis=0)

def eke_map(run):
    out = []
    for f in last(run, "snapshots"):
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values
        uc = 0.5 * (u[..., :-1] + u[..., 1:])                     # xq -> xh
        vc = 0.5 * (v[:, :, :-1, :] + v[:, :, 1:, :])             # yq -> yh
        var = 0.5 * (np.nanvar(uc, axis=0) + np.nanvar(vc, axis=0))   # (zl,y,x)
        hm = np.nanmean(h, axis=0)
        out.append(RHO0 * np.nansum(hm * var, axis=0))
        d.close()
    return np.mean(out, axis=0)

d0 = xr.open_dataset(last("bare", "longmean")[-1], decode_times=False)
rho_l = d0["zl"].values                       # layer target densities
lat, lon = d0["yh"].values, d0["xh"].values
d0.close()
# interface i sits between layers i-1 and i; interior interfaces 1..nz-1
drho = np.diff(rho_l)                          # (nz-1,)

from nw2_common import load_e_rest, ape_map as _ape_corrected   # corrected APE (2026-08-07)
e_rest = load_e_rest(B)
e_bare = emean("bare")

def ape_map(e):
    return _ape_corrected(e, e_rest, drho)

A0, K0 = ape_map(e_bare), eke_map("bare")
res = {}
for r, lab in ORDER:
    res[r] = (ape_map(emean(r)) - A0, eke_map(r) - K0, lab)

w = np.cos(np.deg2rad(lat))[:, None]
print(f"{'run':<10}{'dAPE mean':>12}{'dEKE mean':>12}{'dAPE rms':>11}{'dEKE rms':>11}")
for r, lab in ORDER:
    da, dk, _ = res[r]
    f = lambda x: float(np.nansum(x * w) / np.nansum(w * np.isfinite(x)))
    print(f"{r:<10}{f(da)/1e3:>11.1f}k{f(dk)/1e3:>11.1f}k{np.nanstd(da)/1e3:>10.1f}k{np.nanstd(dk)/1e3:>10.1f}k")
print("  (J/m2, area-weighted; k = 10^3)")

print("\npattern correlation of the RESPONSE between closures (dAPE):")
for a in ("ANN_c1p0", "ANN_c2p0"):
    for b in ("GM400", "GM1600", "MEKE"):
        x, y = res[a][0].ravel(), res[b][0].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        print(f"  r({a:<9} , {b:<7}) = {np.corrcoef(x[m], y[m])[0,1]:+.3f}")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
fig, ax = plt.subplots(2, 6, figsize=(18.0, 6.4), sharex=True, sharey=True, constrained_layout=True)
va = np.nanpercentile(np.abs(np.stack([res[r][0] for r, _ in ORDER])), 99) / 1e6
vk = np.nanpercentile(np.abs(np.stack([res[r][1] for r, _ in ORDER])), 99) / 1e3
for j, (r, lab) in enumerate(ORDER):
    da, dk, _ = res[r]
    p0 = ax[0, j].pcolormesh(lon, lat, da / 1e6, cmap="PuOr_r", vmin=-va, vmax=va, shading="auto", rasterized=True)
    p1 = ax[1, j].pcolormesh(lon, lat, dk / 1e3, cmap="RdBu_r", vmin=-vk, vmax=vk, shading="auto", rasterized=True)
    ax[0, j].set_title(lab, loc="left")
    ax[1, j].set_xlabel("longitude")
ax[0, 0].set_ylabel("$\\Delta$APE\nlatitude"); ax[1, 0].set_ylabel("$\\Delta$EKE\nlatitude")
fig.colorbar(p0, ax=ax[0, :], shrink=0.85, label="$\\Delta$APE [MJ m$^{-2}$]")
fig.colorbar(p1, ax=ax[1, :], shrink=0.85, label="$\\Delta$EKE [kJ m$^{-2}$]")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_ape_eke_maps.png"
fig.savefig(png, dpi=140); print("\nwrote", png)

"""Circulation response: barotropic streamfunction and SSH, for the NW2 Phase-1 set.

The energetics (APE/EKE) say how much a closure changes the energy reservoirs; these say what it
does to the CIRCULATION -- the gyres and the ACC-like channel jet that define this basin. Pavel's
NW2 analysis works in interface structure and SSH (his Fig S3 / Tables S2-S3 compare interface
depths along fixed longitudes against R32), so SSH here is directly comparable to what he reports;
the barotropic streamfunction is added because it is the standard diagnostic for a wind-driven
gyre + circumpolar-channel basin and makes the gyre response legible.

  Psi_bt(x,y) = -sum_{j'<=j} sum_k uh(i,j',k)      [m3/s -> Sv]
  SSH         = e(zi=0)                            [m]

Column 1 shows the unparameterized state (so the reader sees the circulation); the rest are
differences from it. ANN C=2 is omitted -- its first decade carries the thin-layer transient
diagnosed in nw2_spike_diagnosis.py."""
import numpy as np, xarray as xr, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/jul2026_nw2"
NWIN = 5
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"), ("MEKE", "MEKE")]

def fields(run):
    fs = sorted(glob.glob(f"{B}/{run}/output/longmean_*.nc"))[-NWIN:]
    uh, ssh = [], []
    for f in fs:
        d = xr.open_dataset(f, decode_times=False)
        uh.append(np.nansum(d["uh"].mean("time").values, axis=0))   # depth-integrated zonal transport
        ssh.append(d["e"].mean("time").values[0])
        lat, lon = d["yh"].values, d["xh"].values
        d.close()
    U = np.mean(uh, axis=0)
    psi = -np.cumsum(U, axis=0) / 1e6                                # Sv, accumulate northward
    psi = 0.5 * (psi[:, :-1] + psi[:, 1:])                           # xq -> xh
    return psi, np.mean(ssh, axis=0), lat, lon

p0, s0, lat, lon = fields("bare")
res = {r: fields(r)[:2] for r, _ in ORDER}

print(f"{'run':<10}{'Psi range [Sv]':>18}{'d|Psi| rms [Sv]':>18}{'dSSH rms [m]':>14}{'dSSH mean [m]':>15}")
print(f"{'bare':<10}{f'{p0.min():.0f} .. {p0.max():.0f}':>18}{'-':>18}{'-':>14}{'-':>15}")
for r, lab in ORDER:
    p, s = res[r]
    print(f"{r:<10}{f'{p.min():.0f} .. {p.max():.0f}':>18}{np.nanstd(p-p0):>18.2f}"
          f"{np.nanstd(s-s0):>14.3f}{np.nanmean(s-s0):>15.3f}")

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})
n = len(ORDER) + 1
fig, ax = plt.subplots(2, n, figsize=(3.0*n, 6.6), sharex=True, sharey=True, constrained_layout=True)
vp = np.nanpercentile(np.abs(p0), 99); vs = np.nanpercentile(np.abs(s0 - np.nanmean(s0)), 99)
q0 = ax[0, 0].pcolormesh(lon, lat, p0, cmap="RdBu_r", vmin=-vp, vmax=vp, shading="auto", rasterized=True)
plt.colorbar(q0, ax=ax[0, 0], label="$\\Psi_{bt}$ [Sv]")
r0 = ax[1, 0].pcolormesh(lon, lat, s0 - np.nanmean(s0), cmap="RdBu_r", vmin=-vs, vmax=vs, shading="auto", rasterized=True)
plt.colorbar(r0, ax=ax[1, 0], label="SSH anomaly [m]")
ax[0, 0].set_title("no closure", loc="left"); ax[1, 0].set_title("no closure", loc="left")

dp = np.nanpercentile(np.abs(np.stack([res[r][0]-p0 for r,_ in ORDER])), 99)
ds = np.nanpercentile(np.abs(np.stack([res[r][1]-s0 for r,_ in ORDER])), 99)
for j, (r, lab) in enumerate(ORDER, start=1):
    p, s = res[r]
    a0 = ax[0, j].pcolormesh(lon, lat, p-p0, cmap="PuOr_r", vmin=-dp, vmax=dp, shading="auto", rasterized=True)
    a1 = ax[1, j].pcolormesh(lon, lat, s-s0, cmap="PuOr_r", vmin=-ds, vmax=ds, shading="auto", rasterized=True)
    ax[0, j].set_title(lab, loc="left"); ax[1, j].set_title(lab, loc="left")
fig.colorbar(a0, ax=ax[0, 1:], shrink=0.85, label="$\\Delta\\Psi_{bt}$ [Sv]")
fig.colorbar(a1, ax=ax[1, 1:], shrink=0.85, label="$\\Delta$SSH [m]")
ax[0, 0].set_ylabel("barotropic streamfunction\nlatitude"); ax[1, 0].set_ylabel("sea surface height\nlatitude")
for a in ax[1]: a.set_xlabel("longitude")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_circulation.png"
fig.savefig(png, dpi=140); print("\nwrote", png)

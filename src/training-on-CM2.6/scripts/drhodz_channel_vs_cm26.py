"""Dhruv: "is drho/dz actually smaller in the channel than in CM2.6, or did we compute it
differently?" -- a fair question, because the two were NOT computed the same way:

  channel : drdz = np.gradient(rhopot2, z)                       raw centred difference
  CM2.6   : drdz = -(rho0/g) * N_buoyancy^2, and N_buoyancy comes from Nsquared, which applies
            np.maximum(g*drho_dz/rho0, 0.) -- i.e. UNSTABLE/NEUTRAL COLUMNS ARE CLIPPED TO ZERO
            before the square root, then interpolated to centres.

The CM2.6 files also carry `rho`, so here both datasets get the identical raw treatment, and the
two CM2.6 estimates are printed side by side to expose how much the clipping matters. Also compares
|grad_h rho| so we can tell whether the ratio difference comes from the numerator or denominator."""
import numpy as np, xarray as xr

RE, SPONGE_LAT, RHO0_T, G_T = 6.371e6, -30.625, 1025.0, 9.8
B = "/scratch/db194/mom6/feb2026"
Q = [1, 10, 50, 90]

def stats(name, a):
    a = a[np.isfinite(a)]
    q = np.percentile(np.abs(a), Q)
    zero = 100 * np.mean(np.abs(a) < 1e-9)
    print(f"  {name:<34}" + "".join(f"{v:>11.3e}" for v in q) + f"{zero:>9.2f}%")

print(f"  {'|d rho/dz| [kg m-4]':<34}" + "".join(f"{'p'+str(p):>11}" for p in Q) + f"{'~zero':>9}")
for tag, dd in [("p25", 0.25), ("1p0", 1.0)]:
    rd = f"{B}/channel_extra_sponge_slow_woc_{tag}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8"); lat, z = ds["yh"].values, ds["z_l"].values; ds.close()
    keep = np.isfinite(rho) & (np.broadcast_to(lat[None, :, None], rho.shape) < SPONGE_LAT)
    drdz = np.gradient(rho, z, axis=0)
    stats(f"channel {tag}  raw gradient", drdz[keep])
    # the MOM6-side convention: clip unstable to zero, as Nsquared does
    stats(f"channel {tag}  clipped at 0 (as N^2)", np.maximum(-drdz, 0.)[keep])
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    df = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx
    rhox = 0.5 * (df + np.roll(df, -1, axis=-1))
    stats(f"channel {tag}  |d rho/dx| (for scale)", rhox[keep])

d = xr.open_dataset("/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-9/test-0.nc")
p = xr.open_dataset("/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-9/param.nc")
wet = p["wet"].values > 0.5
rho_t = d["rho"].values.astype("f8"); zl = d["zl"].values
drdz_raw = np.gradient(rho_t, zl, axis=0)
drdz_N = -(RHO0_T / G_T) * d["N_buoyancy"].values.astype("f8") ** 2
stats("CM2.6 f9   raw gradient", drdz_raw[wet])
stats("CM2.6 f9   from N_buoyancy (used)", drdz_N[wet])
stats("CM2.6 f9   |d rho/dx| (for scale)", d["rhox"].values.astype("f8")[wet])
d.close(); p.close()

print("\nRatio |grad_h rho| / |d rho/dz| (the slope; larger => Upsilon larger for the same flux):")
print("  computed at the medians above -- see the two CM2.6 rows for the clipping sensitivity.")

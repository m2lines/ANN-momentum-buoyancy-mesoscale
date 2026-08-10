"""Pavel's gold-standard APE validation (Figure-3.ipynb cells 20-21), applied to NW2 R4 GM1600:
compute per-interface APE from the day-37200 SNAPSHOT with nw2_common's functional and compare
interface-by-interface against MOM6's own ocean.stats APE at the same instant."""
import numpy as np, xarray as xr, sys
sys.path.insert(0, "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts")
from nw2_common import G

B = "/scratch/db194/mom6/jul2026_nw2_R4/GM1600/output"
DAY = 37200.0

sn = xr.open_dataset(f"{B}/snapshots_00037050.nc", decode_times=False)
h = sn["h"].sel(time=DAY).values                       # (zl, yh, xh) instantaneous
rho_l = sn["zl"].values
sn.close()
st = xr.open_dataset(f"{B}/ocean.stats.nc", decode_times=False)
i_t = int(np.argmin(np.abs(st["Time"].values - DAY)))
print(f"stats record at Time = {st['Time'].values[i_t]} (target {DAY})")
ape_stats = st["APE"].values[i_t]                      # (16,) Joules per interface
e_rest = -st["H0"].values[i_t]                         # (16,) same instant
st.close()
sg = xr.open_dataset(f"{B}/static.nc", decode_times=False)
area = sg["area_t"].values
D = sg["depth_ocean"].values                           # (yh, xh) positive down
wet = sg["wet"].values > 0.5
sg.close()

# rebuild interfaces from instantaneous h, bottom-up: e[K]=-D, e[k]=e[k+1]+h[k]
K = h.shape[0]
e = np.empty((K + 1,) + h.shape[1:])
e[K] = -np.where(wet, D, np.nan)
for k in range(K - 1, -1, -1):
    e[k] = e[k + 1] + h[k]

drho = np.diff(rho_l)
print(f"\n{'k':>3}{'e_rest':>10}{'mine [J]':>14}{'stats [J]':>14}{'ratio':>9}")
tot_m = tot_s = 0.0
for k in range(1, K):                                  # interior interfaces
    er = e_rest[k]
    hb = np.maximum(e[K] - er, 0.0)
    a = 0.5 * G * drho[k - 1] * ((e[k] - er) ** 2 - hb ** 2)
    mine = np.nansum(a * area * wet)
    tot_m += mine; tot_s += ape_stats[k]
    print(f"{k:>3}{er:>10.1f}{mine:>14.4e}{ape_stats[k]:>14.4e}{mine/ape_stats[k]:>9.4f}")
print(f"{'sum':>3}{'':>10}{tot_m:>14.4e}{tot_s:>14.4e}{tot_m/tot_s:>9.4f}")
print(f"(stats interface 0 [free surface], excluded: {ape_stats[0]:.3e} J; interface 15: {ape_stats[15]:.3e} J)")

# --- and the front, in raw INSTANTANEOUS data: thickness above interface 8 across the jump
lat = xr.open_dataset(f"{B}/snapshots_00037050.nc", decode_times=False)["yh"].values
j0 = int(np.argmin(np.abs(lat - (-55.9))))
print("\nraw instantaneous e8 = e[8] [m] vs latitude at four longitudes (front region):")
lons = [30, 90, 150, 210]
print(f"{'lat':>9}" + "".join(f"{'i_lon='+str(i):>12}" for i in lons))
for j in range(j0, j0 + 14):
    print(f"{lat[j]:>9.3f}" + "".join(f"{e[8, j, i]:>12.1f}" for i in lons))

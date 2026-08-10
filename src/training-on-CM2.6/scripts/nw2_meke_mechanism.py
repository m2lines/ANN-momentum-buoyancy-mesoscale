"""Why does MEKE drain the most APE while sparing resolved EKE? (Pavel's question, 2026-08-10.)

Four diagnostics at 1/4 deg, all from what is already on disk:
  A  the equilibrated effective kappa (0.4*MEKE_Kh from the final restart) -- horizontally it is
     "GM1600 in the ACC, GM~300 in the basins";
  B  per-interface area-mean dAPE vs bare -- vertically MEKE's drain is confined to the upper
     interior (peak 825-1050 m, ~zero below 1600 m; the SQG vertical structure), while GM1600
     spreads over the whole column;
  C  mean-vs-eddy APE split from the last 20 snapshots -- the drain is almost entirely MEAN-state;
     MEKE retains ~77% of eddy APE vs GM1600's ~51%, which is the EKE-sparing in energy form;
  D  per-interface eddy APE of the bare run -- eddy displacements are near equivalent-barotropic
     (rms 15-27 m at every interface), so a depth-tapered kappa only touches the upper part of
     each eddy while GM's depth-uniform kappa grinds the whole eddy column.
Config context: both GM and MEKE runs use FGNV; only MEKE adds KHTH_USE_SQG_STRUCT+SQG_USE_MEKE
(surface-intensified kappa) and the GMwork->MEKE->kappa feedback (MEKE_GMCOEFF=1)."""
import numpy as np, xarray as xr
from nw2_common import emean, load_grid, load_e_rest, ape_map, G, last

B4 = "/scratch/db194/mom6/jul2026_nw2_R4"
KHF = 0.4

# ---------------- A: the equilibrated kappa field
r = xr.open_dataset(f"{B4}/MEKE/RESTART/MOM.res.nc", decode_times=False)
kh = KHF * r["MEKE_Kh"].values.squeeze()
r.close()
sg = xr.open_dataset(f"{B4}/MEKE/output/static.nc", decode_times=False)
wet = sg["wet"].values > 0.5
latg = sg["geolat"].values[:, 0]
D = np.where(wet, sg["depth_ocean"].values, np.nan)
sg.close()
kh = np.where(wet, kh, np.nan)
acc = (latg > -60) & (latg < -40)
bas = latg > -35
print("A) effective kappa = 0.4*MEKE_Kh at final restart [m2/s]:")
for name, m in [("domain", np.ones_like(latg, bool)), ("ACC 40-60S", acc), ("basins N of 35S", bas)]:
    k = kh[m]; k = k[np.isfinite(k)]
    print(f"   {name:<16} median {np.median(k):7.0f}  mean {k.mean():7.0f}  p90 {np.percentile(k,90):7.0f}  "
          f"frac>400: {(k>400).mean():.2f}  frac>1600: {(k>1600).mean():.2f}")

# ---------------- B: per-interface dAPE (which depths does each closure flatten?)
rho_l, lat, lon = load_grid(B4)
drho = np.diff(rho_l)
e_rest = load_e_rest(B4)
w = np.cos(np.deg2rad(lat))[:, None]

def ape_by_k(e):
    ei, er = e[1:-1], e_rest[1:-1][:, None, None]
    hb = np.maximum(e[-1][None] - er, 0.0)
    a = 0.5 * G * drho[:, None, None] * ((ei - er)**2 - hb**2)
    return np.array([np.nansum(a[k] * w) / np.nansum(w * np.isfinite(a[k])) for k in range(14)])

A_b = ape_by_k(emean(B4, "bare"))
runs = {rn: ape_by_k(emean(B4, rn)) - A_b for rn in ("GM1600", "MEKE", "ANN_c1p0")}
print("\nB) area-mean dAPE by interface [kJ/m2] (run - bare); e_rest depth in col 2:")
print(f"   {'k':>3}{'z_rest':>8}" + "".join(f"{rn:>10}" for rn in runs))
for k in range(14):
    print(f"   {k+1:>3}{e_rest[k+1]:>8.0f}" + "".join(f"{runs[rn][k]/1e3:>10.1f}" for rn in runs))

# ---------------- C: mean vs eddy APE split from snapshots (e rebuilt from h, bottom-up)
def snapshots_e(rn):
    es = []
    for f in last(B4, rn, "snapshots"):
        d = xr.open_dataset(f, decode_times=False); h = d["h"].values; d.close()
        for t in range(h.shape[0]):
            e = np.empty((16,) + h.shape[2:]); e[15] = -D
            for k in range(14, -1, -1):
                e[k] = e[k + 1] + h[t, k]
            es.append(e)
    return np.array(es)

def wmean(f):
    return np.nansum(f * w) / np.nansum(w * np.isfinite(f))

print("\nC) APE split over the last 20 snapshots [kJ/m2]: mean-state (MPE) vs eddy (EPE):")
print(f"   {'run':<10}{'MPE':>9}{'EPE':>9}")
base = {}
for rn in ("bare", "GM1600", "MEKE", "ANN_c1p0"):
    es = snapshots_e(rn)
    tot = np.mean([wmean(ape_map(e, e_rest, drho)) for e in es])
    mpe = wmean(ape_map(es.mean(axis=0), e_rest, drho))
    base[rn] = (mpe, tot - mpe)
    print(f"   {rn:<10}{mpe/1e3:>9.1f}{(tot-mpe)/1e3:>9.1f}")
    if rn == "bare":
        var = np.var(es, axis=0)          # keep for D
mb, eb = base["bare"]
print("   deltas vs bare: " + ", ".join(f"{rn}: dMPE {(base[rn][0]-mb)/1e3:+.1f} dEPE {(base[rn][1]-eb)/1e3:+.1f}"
      for rn in ("GM1600", "MEKE", "ANN_c1p0")))

# ---------------- D: per-interface eddy APE of the bare run
print(f"\nD) bare-run eddy APE by interface:\n   {'k':>3}{'z_rest':>8}{'EPE_k [kJ/m2]':>15}{'rms(e_k eddy) [m]':>20}")
for k in range(1, 15):
    epe = 0.5 * G * drho[k - 1] * var[k]
    print(f"   {k:>3}{e_rest[k]:>8.0f}{wmean(epe)/1e3:>15.2f}{np.sqrt(np.nanmean(var[k])):>20.1f}")

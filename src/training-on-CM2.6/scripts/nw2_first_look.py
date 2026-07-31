"""First look at the NW2 Phase-1 set: did the six closures actually produce different solutions,
and in the directions Sec 4.1 would predict? Response is measured against `bare` (NW2 has no
buoyancy closure by default, so bare is a true zero-closure reference). Last 5 windows only, to
drop the adjustment measured in nw2_adjustment_timescale.py."""
import numpy as np, xarray as xr, glob
B = "/scratch/db194/mom6/jul2026_nw2"
RUNS = ["bare", "GM400", "GM1600", "MEKE", "ANN_c1p0", "ANN_c1p5", "ANN_c2p0"]
RHO0 = 1035.0

def load(r, stream, n=5):
    f = sorted(glob.glob(f"{B}/{r}/output/{stream}_*.nc"))[-n:]
    return [xr.open_dataset(x, decode_times=False) for x in f]

res = {}
for r in RUNS:
    lm = load(r, "longmean"); sn = load(r, "snapshots")
    e = np.mean([d["e"].mean("time").values for d in lm], axis=0)
    # uh is m3/s (no rho0); ACC-analogue = max over x of the zonally-... here: depth+meridionally
    # integrated zonal transport through each meridional section, max over sections
    acc = np.mean([float(np.nanmax(np.abs(np.nansum(d["uh"].mean("time").values, axis=(0,1))))/1e6) for d in lm])
    # resolved EKE from snapshot variance within each window
    eke = np.mean([0.5*(float(np.nanmean(np.nanvar(d["u"].values, axis=0)))
                        + float(np.nanmean(np.nanvar(d["v"].values, axis=0)))) for d in sn])
    res[r] = dict(e=e, acc=acc, eke=eke)
    for d in lm+sn: d.close()

b = res["bare"]
print(f"{'run':<10}{'ACC[Sv]':>9}{'EKE':>11}{'EKE %bare':>11}{'rms de vs bare[m]':>19}")
for r in RUNS:
    s = res[r]
    de = np.sqrt(np.nanmean((s["e"]-b["e"])**2))
    print(f"{r:<10}{s['acc']:>9.1f}{s['eke']:>11.4e}{100*s['eke']/b['eke']:>10.0f}%{de:>19.1f}")

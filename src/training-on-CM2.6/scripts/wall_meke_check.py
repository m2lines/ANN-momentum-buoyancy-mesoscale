"""Dhruv's hypothesis (story C): the near-wall APE over-injection may be an indirect effect of
DAMPING RESOLVED EDDIES, not of the ANN's local prediction. Test with existing runs: MEKE damps
eddies far harder than the ANN (EKE 2-11% of no-param vs ~52%), so if story C holds, MEKE's wall
band should over-inject at least as much. Also rank all runs: wall excess vs resolved EKE."""
import numpy as np, xarray as xr, glob, re
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values; w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / (w.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
A_np = ape_map(interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000)[0], R, drho)
yh = A_np[dl(A_np, "yh")].values
target = (ape_map(block(e_tr, 4), R, drho).values - A_np.values) / 1e6
wall = yh < -48.5

# (run label, subdir, cut, EKE % of no-param from prior analyses)
RUNS = [
    ("MEKE khf=0.4",  "tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500, 5.7),
    ("MEKE khf=0.8",  "tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.8", 11500, 1.9),
    ("ANN A LOCAL",   "tau_0.2_cb_1.0_cu_0.0_Aoff_bnd",    11500, 52.0),
    ("ANN B STENCIL", "tau_0.2_cb_1.0_cu_0.0_Boff",        11500, 54.0),
    ("ANN B+clamp",   "tau_0.2_cb_1.0_cu_0.0_Bon",         11500, 58.0),
    ("ANN old off",   "tau_0.2_cb_1.0_cu_0.0_clampoff",    11500, 55.0),
    ("ANN old clamp", "tau_0.2_cb_1.0_cu_0.0_act_relu",    11500, 67.0),
]
print(f"target wall-band dAPE: {np.nanmean(target[wall,:]):+.3f} MJ/m2\n")
print(f"{'run':<15}{'wall dAPE':>10}{'wall excess':>12}{'EKE %np':>9}")
for nm, sub, cut, ekepct in RUNS:
    f = (ape_map(interfaces(f"{P25}/{sub}", cut)[0], R, drho).values - A_np.values) / 1e6
    print(f"{nm:<15}{np.nanmean(f[wall,:]):>10.3f}{np.nanmean((f-target)[wall,:]):>12.3f}{ekepct:>9.1f}")
print("\nStory C predicts: MEKE (most eddy-damped) should sit at or above the ANN's wall excess,")
print("and excess should fall as EKE rises.")

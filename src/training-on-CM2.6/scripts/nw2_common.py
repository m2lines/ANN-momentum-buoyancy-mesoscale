"""Shared NW2 analysis helpers — in particular THE one correct APE functional.

Created 2026-08-07 after the uncorrected APE (horizontal-mean reference, no topography term)
lingered in three figure scripts a week after nw2_ape_check.py had established it was wrong.
Import from here instead of re-deriving APE inline, so a future fix propagates everywhere.

    APE(x,y) = sum_i 0.5 g drho_i [ (e_i - e_rest_i)^2 - max(e_bot - e_rest_i, 0)^2 ]

with e_rest = -H0 (MOM6's own adiabatically-flattened minimum-APE state, from ocean.stats.nc; e is
negative-down, H0 positive-down) and the max(...) term removing the part of the displacement that
is pinned by topography and therefore not available (Perezhogin, notebooks/Figure-3.ipynb). In NW2
~16% of (interface, column) pairs are pinned, and the old form overstated absolute APE by ~215%;
dAPE area-means happen to survive (adiabatic closures conserve layer volume, and the pinned term
cancels in differences) but dAPE maps are ~30% low in rms without the correction."""
import numpy as np, xarray as xr, glob

# NW2 runs with G_EARTH = 10.0 (MOM_parameter_doc.all, both rungs), NOT the 9.8 default.
# Validated 2026-08-10: with 10.0 the per-interface APE of a day-37200 snapshot matches
# ocean.stats.nc APE exactly (with 9.8 every interface was uniformly 0.9800 of stats).
G = 10.0

def last(base, run, stream, n=5):
    """The final n files of a stream, run under `base` (the last windows of the leg)."""
    return sorted(glob.glob(f"{base}/{run}/output/{stream}_*.nc"))[-n:]

def emean(base, run, n=5):
    """Time-mean interface heights over the last n longmean windows (zi, yh, xh)."""
    return np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values
                    for f in last(base, run, "longmean", n)], axis=0)

def load_grid(base, run="bare"):
    """(layer target densities, lat, lon) from a longmean file."""
    d = xr.open_dataset(last(base, run, "longmean")[-1], decode_times=False)
    out = d["zl"].values, d["yh"].values, d["xh"].values
    d.close()
    return out

def load_e_rest(base, run="bare", nrec=200):
    """Reference interfaces e_rest = -H0, averaged over the last nrec ocean.stats records."""
    s = xr.open_dataset(f"{base}/{run}/output/ocean.stats.nc", decode_times=False)
    e_rest = -s["H0"].values[-nrec:].mean(axis=0)
    s.close()
    return e_rest

def eke_map(base, run, n=5):
    """Depth-averaged resolved eddy KE map [m2/s2] from the last n snapshot files.

    Thickness-weighted, eddy = deviation from each file's own time mean (the convention of
    nw2_ladder.py / nw2_vs_truth.py), then averaged across files."""
    out = []
    for f in last(base, run, "snapshots", n):
        d = xr.open_dataset(f, decode_times=False)
        u, v, h = d["u"].values, d["v"].values, d["h"].values
        d.close()
        uc = 0.5 * (u[..., :-1] + u[..., 1:])
        vc = 0.5 * (v[:, :, :-1, :] + v[:, :, 1:, :])
        KE = 0.5 * np.nanmean((uc**2 + vc**2) * h, 0)
        um, vm, hm = np.nanmean(uc, 0), np.nanmean(vc, 0), np.nanmean(h, 0)
        out.append(np.nansum(KE - 0.5 * (um**2 + vm**2) * hm, 0) / np.nansum(hm, 0))
    return np.mean(out, 0)

def ape_map(e, e_rest, drho):
    """Corrected interface-displacement APE [J/m2]: -H0 reference, topography-pinned part removed."""
    ei = e[1:-1]
    er = e_rest[1:-1][:, None, None]
    hb = np.maximum(e[-1][None] - er, 0.0)     # seafloor above the reference -> pinned, unavailable
    return np.nansum(0.5 * G * drho[:, None, None] * ((ei - er) ** 2 - hb ** 2), axis=0)

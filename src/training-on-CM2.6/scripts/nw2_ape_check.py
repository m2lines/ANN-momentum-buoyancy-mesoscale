"""Was the NW2 APE computed properly? Dhruv asked specifically about isopycnals running into
topography. Two errors in the version used so far, both fixed and quantified here.

MY VERSION (nw2_ape_eke_maps.py / nw2_energy_panels.py):
    APE = sum_i 0.5 g drho_i (e_i - R_i)^2 ,  R_i = horizontal mean of the control's interfaces
  (a) the reference is an ad-hoc horizontal mean, not the adiabatically-flattened minimum-APE state;
  (b) NO topography term. Where the seafloor lies ABOVE an interface's reference position the
      interface is pinned to the bottom, and that displacement is NOT available -- it cannot be
      flattened, the topography is in the way. NW2 shoals to ~21 m in places while reference
      interfaces reach 4000 m, so in shallow columns nearly every interface is pinned and
      contributes a large spurious "APE".

PEREZHOGIN'S VERSION (notebooks/Figure-3.ipynb `APE`), which is the correct one:
    hint = e - e_rest ;  hbot = max(e_bottom - e_rest, 0)
    APE  = 0.5 g' [ hint^2 - hbot^2 ]
with e_rest = -H0, H0 being MOM6's own adiabatically-flattened reference interface heights from
ocean.stats.nc. The hbot term removes exactly the pinned, unavailable part.

Note e is NEGATIVE downward in the output (surface ~-0.2 m, bottom to -4000 m) while H0 is POSITIVE
downward, hence the sign flip -- he writes `Z0 = -...H0`.

Since we report RESPONSES, a common reference is used for every run (the control's H0), so that
dAPE is a difference of one functional rather than of two differently-referenced ones."""
import numpy as np, xarray as xr, glob

B = "/scratch/db194/mom6/jul2026_nw2"
G, NWIN = 10.0, 5   # G_EARTH=10.0 in NW2 (validated vs ocean.stats 2026-08-10)
RUNS = ["bare", "GM400", "ANN_c1p0", "GM1600", "ANN_c1p5", "MEKE"]

def last(run, s, n=NWIN): return sorted(glob.glob(f"{B}/{run}/output/{s}_*.nc"))[-n:]
def emean(run):
    return np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values
                    for f in last(run, "longmean")], axis=0)

d0 = xr.open_dataset(last("bare", "longmean")[-1], decode_times=False)
rho_l, lat = d0["zl"].values, d0["yh"].values; d0.close()
drho = np.diff(rho_l)                                    # jumps across interior interfaces

s = xr.open_dataset(f"{B}/bare/output/ocean.stats.nc", decode_times=False)
H0 = s["H0"].values[-200:].mean(axis=0); s.close()       # adiabatically-flattened reference
e_rest = -H0                                             # to e's negative-down convention
print("reference interfaces e_rest [m]:", np.round(e_rest, 1))

e0 = emean("bare")
R_old = np.nanmean(e0[1:-1], axis=(1, 2))                # what I used before

def ape_old(e):
    return np.nansum(0.5*G*drho[:, None, None]*(e[1:-1]-R_old[:, None, None])**2, axis=0)

def ape_new(e):
    """Perezhogin's form: subtract the part pinned by topography."""
    ei = e[1:-1]                                          # interior interfaces
    er = e_rest[1:-1][:, None, None]
    ebot = e[-1][None]                                    # seafloor, negative down
    hint = ei - er
    hbot = np.maximum(ebot - er, 0.0)                     # >0 where the floor is ABOVE the reference
    return np.nansum(0.5*G*drho[:, None, None]*(hint**2 - hbot**2), axis=0)

w = np.cos(np.deg2rad(lat))[:, None]
am = lambda x: float(np.nansum(x*w)/np.nansum(w*np.isfinite(x)))
A_old0, A_new0 = ape_old(e0), ape_new(e0)
frac_pinned = np.nanmean(np.maximum(e0[-1][None] - e_rest[1:-1][:, None, None], 0) > 0)
print(f"\nfraction of (interface, column) pairs pinned by topography: {100*frac_pinned:.1f}%")
print(f"control APE, area mean:  old {am(A_old0)/1e6:8.2f}   corrected {am(A_new0)/1e6:8.2f} MJ/m2"
      f"   -> old overstates by {100*(am(A_old0)/am(A_new0)-1):.0f}%")

print(f"\n{'run':<10}{'dAPE old':>12}{'dAPE corrected':>17}{'change':>10}")
for r in RUNS[1:]:
    e = emean(r)
    o, n = am(ape_old(e)-A_old0)/1e3, am(ape_new(e)-A_new0)/1e3
    print(f"{r:<10}{o:>12.1f}{n:>17.1f}{100*(n-o)/abs(o):>9.0f}%")
print("  (kJ/m2, area-weighted)")

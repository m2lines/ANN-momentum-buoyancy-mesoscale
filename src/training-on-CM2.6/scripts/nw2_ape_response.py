"""dAPE response maps at BOTH NW2 resolutions, with the CORRECTED APE functional.

Supersedes the APE rows of nw2_ape_ke_maps.py / nw2_energy_panels.py / nw2_ape_eke_maps.py, which
were built before nw2_ape_check.py established two errors (2026-07-31): they referenced APE to a
horizontal mean of the control's interfaces rather than -H0, MOM6's adiabatically-flattened
minimum-APE state, and they omitted the topography term. Where the seafloor lies ABOVE an
interface's reference position the interface is pinned to the bottom and that displacement is not
available, so Perezhogin's form (notebooks/Figure-3.ipynb) subtracts it:

    APE = sum_i 0.5 g drho_i [ (e_i - e_rest)^2 - max(e_bot - e_rest, 0)^2 ],   e_rest = -H0

NW2 shoals to ~21 m while reference interfaces reach 4000 m, so ~16% of (interface, column) pairs
are pinned. The correction leaves area-mean dAPE unchanged (the closures are adiabatic, so layer
volume is conserved and the reference-dependent cross term vanishes in the area average; the pinned
term is identical across runs and cancels in any difference) but raises dAPE map rms by ~30%.

Rows are the two rungs, 1/2 deg and 1/4 deg, both branched from day 30001 in the same configuration,
so the columns are directly comparable down the figure. ANN C=2 is not in the 1/4 set."""
import numpy as np, xarray as xr, glob, string
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

NWIN = 5
RUNGS = [("1/2$^\\circ$", "/scratch/db194/mom6/jul2026_nw2"),
         ("1/4$^\\circ$", "/scratch/db194/mom6/jul2026_nw2_R4")]
ORDER = [("GM400", "GM $\\kappa$=400"), ("ANN_c1p0", "ANN $C$=1"),
         ("GM1600", "GM $\\kappa$=1600"), ("ANN_c1p5", "ANN $C$=1.5"), ("MEKE", "MEKE")]

def last(base, r, s, n=NWIN): return sorted(glob.glob(f"{base}/{r}/output/{s}_*.nc"))[-n:]

def rung(base):
    d0 = xr.open_dataset(last(base, "bare", "longmean")[-1], decode_times=False)
    rho_l, lat, lon = d0["zl"].values, d0["yh"].values, d0["xh"].values; d0.close()
    drho = np.diff(rho_l)
    st = xr.open_dataset(f"{base}/bare/output/ocean.stats.nc", decode_times=False)
    e_rest = -st["H0"].values[-200:].mean(axis=0); st.close()

    def emean(r):
        return np.mean([xr.open_dataset(f, decode_times=False)["e"].mean("time").values
                        for f in last(base, r, "longmean")], axis=0)

    from nw2_common import ape_map as _ape_corrected
    def ape(e):
        return _ape_corrected(e, e_rest, drho)

    A0 = ape(emean("bare"))
    return lat, lon, A0, {r: ape(emean(r)) - A0 for r, _ in ORDER}

data = {}
for lab, base in RUNGS:
    data[lab] = rung(base)
    print(f"--- {lab} loaded", flush=True)

print(f"\n{'run':<11}" + "".join(f"{l+' mean':>13}{l+' rms':>12}" for l, _ in RUNGS) + "   [kJ/m2]")
for r, _ in ORDER:
    cells = ""
    for lab, _ in RUNGS:
        lat, lon, A0, res = data[lab]
        w = np.cos(np.deg2rad(lat))[:, None]
        m = float(np.nansum(res[r]*w)/np.nansum(w*np.isfinite(res[r])))
        cells += f"{m/1e3:>13.1f}{np.nanstd(res[r])/1e3:>12.1f}"
    print(f"{r:<11}{cells}")

mpl.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
n = len(ORDER)
fig, ax = plt.subplots(2, n, figsize=(2.45*n, 9.4), constrained_layout=True)
allv = np.concatenate([np.abs(data[l][3][r]).ravel()/1e6 for l, _ in RUNGS for r, _ in ORDER])
allv = allv[np.isfinite(allv)]
lt, vmax = np.percentile(allv, 60), np.percentile(allv, 99.8)
norm = mpl.colors.SymLogNorm(linthresh=lt, vmin=-vmax, vmax=vmax, base=10)
for i, (lab, _) in enumerate(RUNGS):
    lat, lon, A0, res = data[lab]
    for j, (r, rlab) in enumerate(ORDER):
        a = ax[i, j]
        im = a.pcolormesh(lon, lat, res[r]/1e6, norm=norm, cmap=plt.cm.PuOr_r, shading="auto", rasterized=True)
        a.set_aspect("equal")
        lats = [-60, -40, -20, 0, 20, 40, 60]
        a.set_yticks(lats)
        a.set_yticklabels([f"${abs(v)}^\\circ$S" if v < 0 else (f"${v}^\\circ$N" if v > 0 else "$0^\\circ$")
                           for v in lats] if j == 0 else [""]*7)
        a.set_xticks([10, 30, 50])
        a.set_xticklabels([f"${x}^\\circ$E" for x in [10, 30, 50]] if i == 1 else ["", "", ""])
        if i == 0: a.set_title(rlab, loc="left")
        a.text(0.04, 0.975, f"({string.ascii_lowercase[i*n+j]})", transform=a.transAxes,
               va="top", fontsize=10, fontweight="bold")
    ax[i, 0].set_ylabel(f"{lab}\nlatitude")
fig.colorbar(im, ax=ax, orientation="horizontal", extend="both", aspect=60, pad=0.02,
             shrink=0.75).set_label("$\\Delta$APE vs no closure  [MJ m$^{-2}$]", fontsize=10)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/nw2_ape_response.png"
fig.savefig(png, dpi=140, bbox_inches="tight")
fig.savefig("/home/db194/mesoscale_b_ml_parameterization/figures/nw2_ape_response.pdf", dpi=300, bbox_inches="tight")
print("wrote", png, "+ paper PDF")

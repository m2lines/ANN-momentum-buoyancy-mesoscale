"""Choose the GM diffusivities for the NW2 comparison from evidence rather than round numbers.

NeverWorld2's baseline has no thickness diffusion at all (MOM6 defaults: THICKNESSDIFFUSE=False,
KHTH=0), and Pavel's NW2 comparisons are momentum closures (ZB2020, Yankovsky), so there is no
published GM value to inherit. We therefore set kappa from the ANN's own implied diffusivity in
this configuration, so the diffusive runs bracket the scheme rather than being arbitrary.

Method: MOM6 hands the thickness-diffusion routine a volume streamfunction, and for GM that
streamfunction is Sfn = -kappa * s * dy. Diagnosing the ANN run's realized Sfn and the isopycnal
slope on the same faces gives the diffusivity a GM run would need to move the same transport,

    kappa_equiv = |Sfn| / (|s| * dy)

evaluated where the transport actually lives (we report the transport-weighted value, which is what
a bulk comparison should match, alongside the plain median).

Then GM_lo = kappa_equiv/2 and GM_hi = 2*kappa_equiv, rounded to a tidy value: a pair that brackets
the ANN's strength so the claim becomes 'no diffusive strength reproduces the ANN's character'
rather than 'these two particular kappas do not'."""
import numpy as np, xarray as xr, glob, sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "/scratch/db194/mom6/jul2026_nw2/smoke"
RHO0 = 1035.0

lm = sorted(glob.glob(f"{RUN}/output/longmean_*.nc")) or sorted(glob.glob(f"{RUN}/longmean_*.nc"))
if not lm:
    sys.exit(f"no longmean output in {RUN} yet")
d = xr.concat([xr.open_dataset(f, decode_times=False) for f in lm], dim="time")
t = "time" if "time" in d.dims else None
g = lambda v: (d[v].mean(t) if t else d[v]).values

e = g("e")                                   # interface heights (zi, yh, xh)
uh = g("uh")                                 # zonal thickness transport (zl, yh, xq)
lat = d["yh"].values
# grid metrics: NW2 is a regular lat-lon grid
dlon = float(d["xh"][1] - d["xh"][0]); dlat = float(d["yh"][1] - d["yh"][0])
RE = 6.371e6
dy = np.deg2rad(dlat) * RE
dx = np.deg2rad(dlon) * RE * np.cos(np.deg2rad(lat))[:, None]

# isopycnal slope at u-faces from the interface field
dedx = (np.roll(e, -1, axis=-1) - e) / dx[None]          # (zi, yh, xh) -> u faces
s = np.abs(dedx[1:-1])                                    # interior interfaces

# the ANN's realized eddy transport per unit width, from uh (m3/s -> m2/s by dividing by dy)
Ups = np.abs(uh[:, :, :-1]) / (RHO0 * dy) if uh.shape[-1] == e.shape[-1] + 1 else np.abs(uh) / (RHO0 * dy)
n = min(Ups.shape[0], s.shape[0])
Ups, s = Ups[:n], s[:n]

ok = np.isfinite(Ups) & np.isfinite(s) & (s > 1e-6) & (Ups > 0)
kappa = Ups[ok] / s[ok]
w = Ups[ok]                                               # transport weighting
kw = float(np.sum(kappa * w) / np.sum(w))
print(f"n = {ok.sum():,} u-faces with a resolvable slope")
for q in (25, 50, 75, 90):
    print(f"  kappa_equiv p{q:<3} {np.percentile(kappa, q):8.1f} m2/s")
print(f"  transport-weighted mean {kw:8.1f} m2/s   <- the value a bulk GM comparison should match")

def tidy(x):
    for c in (50, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000):
        if c >= x: return c
    return int(round(x / 500) * 500)
lo, hi = tidy(kw / 2), tidy(kw * 2)
print(f"\nproposed bracket:  GM_lo = {lo} m2/s   GM_hi = {hi} m2/s   (ANN-equivalent ~{kw:.0f})")

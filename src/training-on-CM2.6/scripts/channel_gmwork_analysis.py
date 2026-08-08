"""Task A validation + deliverables (PLAN 2026-08-07 cont 3): the ONLINE applied-release vertex.

The five _gmwork continuations wrote 100-day means over a fresh 2000-day window of:
  GMwork            -- thickness_diffuse's exact applied APE-extraction map [W/m2], every timestep
  GM_sfn(_unlim)_x/y -- total volume streamfunction after/before the downstream machinery
  meso_sfn_unlim_u/v -- the ANN's requested velocity-scale transport (ANN runs; KHTH=0 there, so
                        the GM_sfn fields carry the ANN contribution alone)
  MEKE_KH            -- MEKE's diffusivity (MEKE run)

Validation sequence (ordered, per the task):
  V1  the GM run's GMwork must be single-signed -- a strictly downgradient closure can only
      extract APE; this pins the sign convention for every other run.
  V2  the ANN 1/4-deg GMwork pattern should broadly correspond to the truth-diagnosed deployed
      release (Task B: channel_release/woc_p0625_factor4.nc, a_tap time-mean).
Deliverables:
  (a) the three-way applied-release map figure: truth-diagnosed vs ANN vs GM (vs MEKE), 1/4 deg;
  (b) ladder numbers: domain-integrated applied release, ANN at 1/4, 1/2, 1 deg;
  (c) the ONLINE REMIT: realized/requested transport (GM_sfn vs GM_sfn_unlim) per run, to set
      against the offline 15->10% APE-release remit.
Sponge bands masked from all statistics (restoring, not free dynamics)."""
import numpy as np, xarray as xr, glob
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt

B = "/scratch/db194/mom6/feb2026"
TB = "/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/channel_release/woc_p0625_factor4.nc"
SPONGE_LAT = -30.625
RUNS = {
    "ANN p25":  "channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_neutral_gmwork",
    "ANN p5":   "channel_extra_sponge_slow_woc_p5/tau_0.2_cb_1.0_cu_0.0_neutral_gmwork",
    "ANN 1p0":  "channel_extra_sponge_slow_woc_1p0/tau_0.2_cb_1.0_cu_0.0_neutral_gmwork",
    "GM kh1000":"channel_extra_sponge_slow_woc_p25/tau_0.2_cb_0.0_cu_0.0_GM_kh1000_gmwork",
    "MEKE 0.4": "channel_extra_sponge_slow_woc_p25/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4_gmwork",
}

def load(run):
    f = sorted(glob.glob(f"{B}/{run}/output/gmwork_*.nc") + glob.glob(f"{B}/{run}/gmwork_*.nc"))
    return xr.open_mfdataset(f, combine="by_coords", decode_times=False) if f else None

data = {k: load(v) for k, v in RUNS.items()}

# ---------------- V1: sign convention from the GM run
g = data["GM kh1000"]["GMwork"].mean("Time").values
lat = data["GM kh1000"]["yh"].values
keep = lat < SPONGE_LAT
gk = g[keep]
pos, neg = float(np.nansum(gk[gk > 0])), float(np.nansum(gk[gk < 0]))
print("V1  GM kh1000 GMwork (sponge masked):")
print(f"    positive part {pos:.3e}  negative part {neg:.3e}  |neg|/pos = {abs(neg)/max(pos,1e-30):.4f}")
print(f"    -> single-signed: {'YES' if min(pos, abs(neg))/max(pos, abs(neg)) < 0.05 else 'NO (investigate before use)'}")
sgn = 1.0 if pos > abs(neg) else -1.0
print(f"    convention: GMwork {'>' if sgn>0 else '<'} 0 = APE extraction\n")

# ---------------- V2: ANN p25 pattern vs the truth-diagnosed deployed release (Task B)
tb = xr.open_dataset(TB, decode_times=False)
# Task B stores depth-integrated release maps directly: ape_diag (truth), ape_pred (offline ANN),
# ape_gm (offline GM analogue), all (yh, xh) on the factor-4 coarse grid + a wet mask.
msk = tb["mask2d"].values > 0.5
a_diag = np.where(msk, tb["ape_diag"].values, np.nan)
a_pred = np.where(msk, tb["ape_pred"].values, np.nan)
w_ann = data["ANN p25"]["GMwork"].mean("Time").values
ylat = data["ANN p25"]["yh"].values
k2 = ylat < SPONGE_LAT
# Task B grid should be the same factor-4 coarse grid; align defensively by shape
if a_diag.shape == w_ann.shape:
    x, y = (sgn*w_ann)[k2].ravel(), a_diag[k2].ravel()
    m = np.isfinite(x) & np.isfinite(y)
    r = np.corrcoef(x[m], y[m])[0, 1]
    amp = float(np.nansum(x[m]) / np.nansum(y[m]))
    print(f"V2  ANN p25 GMwork vs truth-diagnosed release: pattern r = {r:.3f}, "
          f"integral ratio online/diag = {amp:.3f}")
else:
    print(f"V2  SHAPE MISMATCH GMwork {w_ann.shape} vs diag {a_diag.shape} -- align before comparing")
    r, amp = np.nan, np.nan

# ---------------- (b) ladder numbers
print("\n(b) domain-integrated applied APE extraction (sponge masked), sign-fixed:")
RE = 6.371e6
for k in ("ANN p25", "ANN p5", "ANN 1p0", "GM kh1000", "MEKE 0.4"):
    d = data[k]
    wm = d["GMwork"].mean("Time").values
    la = d["yh"].values; lo = d["xh"].values
    dd = float(lo[1]-lo[0])
    dyy = np.deg2rad(dd)*RE; dxx = dyy*np.cos(np.deg2rad(la))[:, None]
    kk = la < SPONGE_LAT
    tot = np.nansum((sgn*wm*dxx*dyy)[kk])
    print(f"    {k:<10} {tot/1e9:8.2f} GW")

# ---------------- (c) online remit
print("\n(c) online remit  |GM_sfn| / |GM_sfn_unlim|  (transport-weighted, interior interfaces, sponge masked):")
for k in ("ANN p25", "ANN p5", "ANN 1p0", "GM kh1000", "MEKE 0.4"):
    d = data[k]
    sx  = d["GM_sfn_x"].mean("Time").values[1:-1]
    sxu = d["GM_sfn_unlim_x"].mean("Time").values[1:-1]
    la = d["yh"].values; kk = la < SPONGE_LAT
    a, b = np.abs(sx[:, kk, :]), np.abs(sxu[:, kk, :])
    m = np.isfinite(a) & np.isfinite(b) & (b > 0)
    print(f"    {k:<10} Sum|realized|/Sum|requested| = {np.nansum(a[m])/np.nansum(b[m]):.3f}   "
          f"median ratio = {np.nanmedian(a[m]/b[m]):.3f}")

# ---------------- (a) three-way figure at 1/4 deg
fig, ax = plt.subplots(1, 5, figsize=(18.5, 3.6), sharex=True, sharey=True, constrained_layout=True)
fields = [("truth-diagnosed (deployed operator)", a_diag),
          ("ANN offline (deployed operator)", a_pred),
          ("ANN $C$=1 online (GMwork)", sgn*w_ann),
          ("GM $\\kappa$=1000 online", sgn*data["GM kh1000"]["GMwork"].mean("Time").values),
          ("MEKE khf=0.4 online", sgn*data["MEKE 0.4"]["GMwork"].mean("Time").values)]
vals = np.concatenate([np.abs(f[k2]).ravel() for _, f in fields if f is not None])
vm = np.nanpercentile(vals[np.isfinite(vals)], 99)
for a_, (ttl, f) in zip(ax, fields):
    if f is None:
        a_.set_visible(False); continue
    lo = data["ANN p25"]["xh"].values
    pc = a_.pcolormesh(lo, ylat, f, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    a_.axhspan(SPONGE_LAT, ylat[-1], color="gray", alpha=0.25)
    a_.set_title(ttl, fontsize=9.5, loc="left"); a_.set_xlabel("lon")
ax[0].set_ylabel("lat")
fig.colorbar(pc, ax=ax, shrink=0.85, label="applied APE extraction  [W m$^{-2}$]")
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/channel_applied_release.png"
fig.savefig(png, dpi=150); print("\nwrote", png)

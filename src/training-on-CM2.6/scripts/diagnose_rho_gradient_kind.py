"""Tier-1 of the density-gradient "kind" diagnostic (see ../PLAN.md M1 audit item).

The offline training builds its density-gradient inputs as the horizontal gradient of
SURFACE-REFERENCED potential density (sigma0 = gsw.rho(S,T,p=0); state_functions.py:1833,
1990). Online, MOM6's calc_isoneutral_slopes evaluates the EOS density derivatives at the
LOCAL interface pressure (MOM_isopycnal_slopes.F90:336) -> the horizontal density gradient
is taken in a locally-referenced (neutral) framework. The two agree near the surface but
diverge with depth through thermobaricity (alpha, beta depend on pressure). This script
scopes that divergence -- magnitude and direction -- as a function of depth, model-free, on
WOA13 climatology, before deciding whether a retrain on locally-referenced gradients is needed.

Design: isolate the ONE factor in question (reference pressure) by computing both density
fields with the SAME EOS (gsw) and the SAME finite-difference scheme, changing only p:
  rho_0 = gsw.rho(S, T, 0)            <- offline form (sigma0), the gradient training sees
  rho_p = gsw.rho(S, T, p=depth)      <- online form; p uniform per level so grad rho_p
                                          = alpha(p) grad T + beta(p) grad S (pure thermobaric
                                          modulation, no spurious compression term since
                                          grad_h p = 0 at a level)
We pass practical-S / potential-T straight into gsw, mirroring the offline pipeline's own
convention (state_functions.py:1821-1822); since BOTH sides use it, it cancels and the
comparison isolates the reference-pressure effect. WOA's smooth 1deg fields make the
secant (field-difference) and tangent (analytic alpha/beta) forms equivalent, so the
field-gradient on both sides also isolates the same effect (the "secant vs tangent" subtlety
is second-order here).

Per ocean point and level we report the magnitude ratio |grad rho_p| / |grad rho_0| and the
rotation angle between the two gradient vectors; aggregated by depth, gradient-magnitude
weighted (strong gradients carry the flux, so they dominate the relevant statistic).

  TDATA / SDATA  WOA13 ptemp / salinity netcdf  [default OM4 INPUT copies]
  OUT            output dir for npz + figure     [default scratch rho_grad_kind/]
"""
import os
import numpy as np
import gsw
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = '/scratch/db194/OM4/test_ANN_20Dec_torch_5yr/INPUT'
TDATA = os.environ.get('TDATA', f'{INPUT}/woa13_decav_ptemp_monthly_fulldepth_01.nc')
SDATA = os.environ.get('SDATA', f'{INPUT}/woa13_decav_s_monthly_fulldepth_01.nc')
OUT = os.path.expandvars(os.environ.get('OUT', '/scratch/$USER/mom6/CM26_ML_models/FGR3/rho_grad_kind'))
MAP_DEPTH = float(os.environ.get('MAP_DEPTH', 1000.))   # depth (m) for the rotation map panel
FLUX_BAND = float(os.environ.get('FLUX_BAND', 1500.))   # depth (m) above which the rho-flux matters
R_EARTH = 6371.0e3
os.makedirs(OUT, exist_ok=True)

# annual-mean climatology (average the 12 monthly objectively-analyzed fields)
T = xr.open_dataset(TDATA, decode_times=False)['ptemp_an'].mean('time')   # (depth, lat, lon), degC
S = xr.open_dataset(SDATA, decode_times=False)['s_an'].mean('time')       # practical salinity
depth = T.depth.values                                                    # m, length 102
lat = T.lat.values
lon = T.lon.values
t = T.values
s = S.values
print(f'WOA13 annual mean: {t.shape} (depth,lat,lon); depth {depth[0]:.0f}-{depth[-1]:.0f} m', flush=True)

# density fields: p=0 (offline) vs p=local-depth-in-dbar (online), uniform p per level
rho0 = gsw.rho(s, t, 0.0)                                  # (depth, lat, lon)
rho_p = gsw.rho(s, t, depth[:, None, None])               # broadcast p over each level


def hgrad(f):
    """Horizontal gradient (d/dx, d/dy) of a (depth,lat,lon) field on the 1deg sphere.
    Centered, periodic in longitude; metric factors applied so directions are physical."""
    dlam = np.deg2rad(np.gradient(lon))                    # ~uniform 1deg
    dphi = np.deg2rad(np.gradient(lat))
    cosphi = np.cos(np.deg2rad(lat))[None, :, None]
    dfdlam = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2 * np.deg2rad(1.0))   # periodic lon
    dfdphi = np.gradient(f, np.deg2rad(lat), axis=1)                                     # lat (non-periodic)
    gx = dfdlam / (R_EARTH * cosphi)
    gy = dfdphi / R_EARTH
    return gx, gy


gx0, gy0 = hgrad(rho0)
gxp, gyp = hgrad(rho_p)
mag0 = np.sqrt(gx0**2 + gy0**2)
magp = np.sqrt(gxp**2 + gyp**2)
dot = gx0 * gxp + gy0 * gyp
cos_ang = np.clip(dot / (mag0 * magp + 1e-30), -1.0, 1.0)
angle = np.rad2deg(np.arccos(cos_ang))                    # rotation of the gradient direction
ratio = magp / (mag0 + 1e-30)                             # rescaling of the gradient magnitude


def wstats(x, w, mask):
    """gradient-weighted mean + unweighted median over masked points."""
    x, w = x[mask], w[mask]
    if x.size < 50:
        return np.nan, np.nan
    return float(np.sum(w * x) / np.sum(w)), float(np.median(x))


prof = {k: [] for k in ['depth', 'n', 'ratio_w', 'ratio_med', 'angle_w', 'angle_med']}
print('\n  depth     N      |grad_p|/|grad_0|        angle(grad_p, grad_0) deg', flush=True)
print('   (m)            wmean   median            wmean   median', flush=True)
for k in range(len(depth)):
    m = np.isfinite(mag0[k]) & np.isfinite(magp[k]) & (mag0[k] > 0)
    rw, rm = wstats(ratio[k], mag0[k], m)
    aw, am = wstats(angle[k], mag0[k], m)
    prof['depth'].append(depth[k]); prof['n'].append(int(m.sum()))
    prof['ratio_w'].append(rw); prof['ratio_med'].append(rm)
    prof['angle_w'].append(aw); prof['angle_med'].append(am)
    if depth[k] <= 2000 or k % 5 == 0:
        band = ' <- flux band' if depth[k] <= FLUX_BAND else ''
        print(f'  {depth[k]:6.0f}  {int(m.sum()):6d}    {rw:5.3f}   {rm:5.3f}    '
              f'      {aw:5.2f}   {am:5.2f}{band}', flush=True)

prof = {k: np.array(v) for k, v in prof.items()}
np.savez(f'{OUT}/profile.npz', **prof)

# ---- figure: ratio vs depth, angle vs depth, rotation map at MAP_DEPTH ----
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
d = prof['depth']
ax[0].plot(prof['ratio_w'], d, '-', label='grad-weighted mean')
ax[0].plot(prof['ratio_med'], d, '--', color='gray', label='median')
ax[0].axvline(1.0, color='k', lw=0.6)
ax[0].axhspan(0, FLUX_BAND, color='C2', alpha=0.08)
ax[0].set_xlabel(r'$|\nabla\rho_{local}| / |\nabla\sigma_0|$'); ax[0].set_ylabel('depth (m)')
ax[0].set_title('magnitude rescaling'); ax[0].legend(fontsize=8)

ax[1].plot(prof['angle_w'], d, '-', label='grad-weighted mean')
ax[1].plot(prof['angle_med'], d, '--', color='gray', label='median')
ax[1].axhspan(0, FLUX_BAND, color='C2', alpha=0.08, label=f'flux band (<{FLUX_BAND:.0f} m)')
ax[1].set_xlabel(r'angle$(\nabla\rho_{local}, \nabla\sigma_0)$ (deg)')
ax[1].set_title('direction rotation'); ax[1].legend(fontsize=8)
for a in ax[:2]:
    a.set_ylim(depth.max(), 0); a.grid(alpha=0.3)

kmap = int(np.argmin(np.abs(depth - MAP_DEPTH)))
im = ax[2].pcolormesh(lon, lat, angle[kmap], vmin=0, vmax=np.nanpercentile(angle[kmap], 98),
                      cmap='magma', shading='auto')
ax[2].set_title(f'rotation angle (deg) at {depth[kmap]:.0f} m')
ax[2].set_xlabel('lon'); ax[2].set_ylabel('lat')
fig.colorbar(im, ax=ax[2], shrink=0.8)
fig.suptitle(r'Density-gradient kind: $\nabla\sigma_0$ (offline) vs local-pressure $\nabla\rho$ (online), WOA13')
fig.tight_layout()
fig.savefig(f'{OUT}/rho_grad_kind.png', dpi=130)
print(f'\nwrote {OUT}/profile.npz and {OUT}/rho_grad_kind.png', flush=True)

# ---- headline numbers for the decision ----
fb = prof['depth'] <= FLUX_BAND
deep = prof['depth'] >= 2000
print('\n=== headline (gradient-weighted) ===', flush=True)
print(f'  flux band (<{FLUX_BAND:.0f} m):  ratio {np.nanmean(prof["ratio_w"][fb]):.3f}   '
      f'angle {np.nanmean(prof["angle_w"][fb]):.2f} deg', flush=True)
print(f'  deep (>2000 m):       ratio {np.nanmean(prof["ratio_w"][deep]):.3f}   '
      f'angle {np.nanmean(prof["angle_w"][deep]):.2f} deg', flush=True)

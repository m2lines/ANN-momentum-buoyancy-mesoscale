"""Truth-diagnosed (and offline-predicted) DEPLOYED APE release for the channel on the 1/4-deg
grid -- Task B of the online forcing/APE-release metrics (PLAN.md 2026-08-07 cont 3).

Chain per truth snapshot (woc_p0625 no-param run, 40 snaps, days 10100-14000):
fine state (rhopot2, u, v from prog_z) -> CM2.6 coarsen/filter pipeline (factor 4 = 1/4 deg,
FGR 3) -> diagnosed subfilter flux F_diag + ANN prediction F_pred (canonical EXP_neutral_all4,
the same weights the online woc runs load) -> Upsilon = F/|grad3 rho| per in-plane component
(LOCAL_GRAD, exactly the old canonical binary's conversion; no 50 m zeroing since USE_EOS=True)
-> FGNV solve (gamma=1, C_MIN=0.01, modal cg1, bathymetry-aware) -> a_tap = Ups_tap.grad_h(rho).
FGNV params confirmed against the woc p25 neutral run's MOM_parameter_doc.all (2026-08-07).

Outputs (time means): map of int a dz (diag/pred), zonal-mean a(y,z) and tapered Upsilon(y,z)
(diag/pred; Upsilon_y is the piece comparable to the online runs' vhGM-derived streamfunction),
and the area-integrated release ratio pred/diag -- the channel analogue of the CM2.6 deployed
ratio (1.14 at 0.9 deg).

Masking: the northern sponge band is zeroed in the wet mask at the fine level (skill convention,
eval_channel_offline.py) and the SURFACE wet mask is eroded 2 cells horizontally for all
accumulations -- drops the filter rim at the sponge edge and walls. Bathymetric (seamount) edges
are NOT masked: a_tap is a pointwise product and near-topo release is real (CM2.6 lesson:
masking the APE budget biased the ratio +0.12).

Sign convention: code fluxes are -F^rho (training convention), so a_tap > 0 = APE release,
identical to the CM2.6 deployed analysis (diag_fgnv_forcing.py).

GM column (2026-08-07, Dhruv): a constant-kappa GM with kappa fitted by least squares on the
DIAGNOSED FLUX over the full domain (volume-weighted, all snapshots) -- the same objective the
ANN trains on, one parameter vs 10^5. kappa* = <F.grad_h rho>_V / <|grad_h rho|^2>_V (code sign:
downgradient => F_code = +kappa grad rho => kappa* > 0). Everything downstream of the flux
(Upsilon, FGNV, a) is LINEAR in the flux, so GM release is accumulated at kappa=1 in the same
pass and scaled by kappa* at the end -- exact, no second pass.

Env: NMAX=<n> limits snapshots (smoke test); default all.
"""
import os, sys, glob
import numpy as np, xarray as xr
from scipy.linalg import solve_banded, eigh_tridiagonal
from scipy.ndimage import binary_erosion

sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.cm26 import DatasetCM26, create_grid, discard_land, mask_from_nans
from helpers.operators import CoarsenKochkovMinMax, Filtering
from helpers.ann_tools import import_ANN

RUN = '/scratch/db194/mom6/feb2026/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0/output'
MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
OUT = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/channel_release/woc_p0625_factor4.nc'
FACTOR, FGR, STENCIL = 4, 3, 3
RHO0, G = 1035.0, 9.8                       # run's RHO_0 / G_EARTH
GAMMA, C_MIN = 1.0, 0.01                    # FGNV_FILTER_SCALE / FGNV_C_MIN
N2_FLOOR = (1e-15 * 7.2921e-5) ** 2         # FGNV_STRAT_FLOOR convention (negligible)
NMAX = int(os.environ.get('NMAX', 0))

# ----- verified FGNV solver + modal cg1 (copied from diag_fgnv_taper.py, five-way self-tested) --
def fgnv_solve(psi, hN2, c2_h, gamma):
    M = len(hN2)
    ab = np.zeros((3, M))
    ab[1, :] = hN2 + c2_h[:M] + c2_h[1:]
    ab[0, 1:] = -c2_h[1:M]
    ab[2, :-1] = -c2_h[1:M]
    rhs = (1.0 + gamma) * (hN2[:, None] if psi.ndim > 1 else hN2) * psi
    return solve_banded((1, 1), ab, rhs)

def cg1_modal(N2, h):
    """First-baroclinic wave speed from w'' = -(N^2/c^2) w, w=0 at both ends.
    N2: (M,) at interior nodes; h: (M+1,) layer thicknesses. Symmetric tridiag eigenproblem."""
    hz = 0.5 * (h[:-1] + h[1:])
    B = np.maximum(N2, 1e-14) * hz
    d = (1. / h[:-1] + 1. / h[1:]) / B
    e = -1. / (h[1:-1] * np.sqrt(B[:-1] * B[1:]))
    try:
        lam = eigh_tridiagonal(d, e, select='i', select_range=(0, 0), eigvals_only=True)[0]
    except Exception:
        A = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        lam = np.linalg.eigvalsh(A)[0]
    return 1.0 / np.sqrt(max(lam, 1e-30))
# ------------------------------------------------------------------------------------------------

# ----- fine-grid reader (adapter from eval_channel_offline.py; rhopot2 instead of temp) ---------
files = sorted(glob.glob(f'{RUN}/prog_z_0*.nc'))
_p0 = xr.open_dataset(files[0], decode_times=False)
# the woc truth wrote no ocean_geometry.nc; the non-woc p0625 truth has the identical grid
GEOM = '/scratch/db194/mom6/feb2026/channel_extra_sponge_slow_p0625/tau_0.2_cb_0.0_cu_0.0/output/ocean_geometry.nc'
geom = xr.open_dataset(GEOM, decode_times=False).rename(
    {'lonh': 'xh', 'lath': 'yh', 'lonq': 'xq', 'latq': 'yq'})
geom = geom.isel(xq=slice(1, None), yq=slice(1, None))   # symmetric-memory extra SW face

_ref = _p0.isel(Time=[0]).rename({'Time': 'time', 'z_l': 'zl', 'z_i': 'zi'}
                                 ).isel(xq=slice(1, None), yq=slice(1, None))
param = xr.Dataset()
for k in ['dxT', 'dyT', 'dxBu', 'dyBu', 'dxCu', 'dyCv', 'dyCu', 'dxCv', 'geolat', 'geolon', 'wet']:
    if k in geom:
        param[k] = (geom[k].dims, geom[k].values)
param = param.assign_coords(xh=_ref.xh, yh=_ref.yh, xq=_ref.xq, yq=_ref.yq,
                            zl=_ref.zl, zi=_ref.zi)
param['wet'] = mask_from_nans(_ref.rhopot2.isel(time=0, drop=True)).transpose('zl', 'yh', 'xh')
idamp = xr.open_dataset(f'{os.path.dirname(RUN)}/INPUT/sponge_damping_NI640_NJ368_relax_360_days.nc',
                        decode_times=False).Idamp.isel(z1=0)
sponge = xr.DataArray(idamp.values > 0, dims=('yh', 'xh'))
param['wet'] = param['wet'].where(~sponge, 0.)
_grid = create_grid(param)
param['wet_u'] = discard_land(_grid.interp(param['wet'], 'X'))
param['wet_v'] = discard_land(_grid.interp(param['wet'], 'Y'))
param['wet_c'] = discard_land(_grid.interp(param['wet'], ['X', 'Y']))
param['wet_w'] = discard_land(_grid.interp(param['wet'].chunk({'zl': -1}), 'Z'))
for q, ax in [('u', 'X'), ('v', 'Y'), ('w', ['X', 'Y'])]:
    param[f'geolon_{q}'] = _grid.interp(param['geolon'], ax)
    param[f'geolat_{q}'] = _grid.interp(param['geolat'], ax)
print('fine param ready | wet frac', float((param.wet > 0.5).mean()), flush=True)

ann = import_ANN(MODEL)
zl = _ref.zl.values
NZ = len(zl)
DZL = np.gradient(zl)

# accumulators (filled on first snapshot once coarse shape is known)
ACC = None
cg1_map, col_meta = None, None
nt = 0

snaps = [(fn, it) for fn in files
         for it in range(xr.open_dataset(fn, decode_times=False).sizes['Time'])]
if NMAX:
    snaps = snaps[:NMAX]
print(f'{len(snaps)} snapshots to process', flush=True)

for fn, it in snaps:
    prog = xr.open_dataset(fn, decode_times=False).isel(Time=[it]).rename(
        {'Time': 'time', 'z_l': 'zl', 'z_i': 'zi'}).isel(xq=slice(1, None), yq=slice(1, None))
    data = xr.Dataset()
    data['u'] = prog.u.fillna(0.)
    data['v'] = prog.v.fillna(0.)
    data['rho'] = prog.rhopot2.fillna(0.)
    data = data.assign_coords(xh=param.xh, yh=param.yh, xq=param.xq, yq=param.yq,
                              zl=param.zl, zi=param.zi)

    fine = DatasetCM26(data, param)
    coarse = fine.compute_subfilter_forcing(factor=FACTOR, FGR_multiplier=FGR,
                                            coarsening=CoarsenKochkovMinMax(),
                                            filtering=Filtering(), add_rho_fluxes=True)
    feat, feat_const = coarse.state.prepare_features()
    feat = xr.merge([feat, feat_const])
    for v in ('sh_xx', 'sh_xy_h', 'rel_vort_h', 'N_buoyancy'):
        if v in feat and 'time' not in feat[v].dims:
            feat[v] = feat[v].broadcast_like(feat['rhox'])   # exact for the 1-snapshot loop
    ready = DatasetCM26(feat, coarse.param)
    one = ready.predict_ANN_rho(ann, stencil_size=STENCIL).data.isel(time=0)

    rhox, rhoy = one.rhox.astype('float64'), one.rhoy.astype('float64')
    N2 = (one.N_buoyancy.astype('float64') ** 2).clip(min=N2_FLOOR)
    rhoz = -(RHO0 / G) * N2
    magx = np.sqrt(rhox ** 2 + rhoz ** 2)
    magy = np.sqrt(rhoy ** 2 + rhoz ** 2)
    U = {'px': (one.Fx_pred.astype('float64') / (magx + 1e-300)).values,
         'py': (one.Fy_pred.astype('float64') / (magy + 1e-300)).values,
         'dx': (one.Fx.astype('float64') / (magx + 1e-300)).values,
         'dy': (one.Fy.astype('float64') / (magy + 1e-300)).values,
         'gx': (rhox / (magx + 1e-300)).values,      # GM at kappa=1: F_code = grad_h rho
         'gy': (rhoy / (magy + 1e-300)).values}
    N2v = N2.values

    if ACC is None:
        ny, nx = N2v.shape[1], N2v.shape[2]
        cwet = coarse.param.wet.values > 0.5                     # (zl, y, x)
        m2 = binary_erosion(cwet[0], iterations=2)               # surface mask, sponge/wall rim off
        area = (coarse.param.dxT * coarse.param.dyT).values
        ACC = {k: np.zeros((ny, nx)) for k in ('ape_d', 'ape_p', 'ape_g')}
        ACC.update({k: np.zeros((NZ, ny)) for k in
                    ('sa_d', 'sa_p', 'sa_g', 'su_d', 'su_p', 'su_g', 'sn')})
        ACC.update(int_d=0., int_p=0., int_g=0., fit_fg=0., fit_gg=0.)
        # cg1 + column metadata from the first snapshot (cg1 is a slow function of state)
        print(f'coarse {ny}x{nx}; cg1 for {int(m2.sum())}-ish columns...', flush=True)
        col_meta = {}
        cg1_map = np.full((ny, nx), np.nan)
        wetcol = np.isfinite(U['dx']) & np.isfinite(N2v)
        for j in range(ny):
            for i in range(nx):
                w = wetcol[:, j, i]
                M = int(w.sum())
                if M < 8 or not w[:M].all():
                    continue
                z = zl[:M]
                zb = z[-1] + 0.5 * (z[-1] - z[-2])
                h = np.diff(np.concatenate([[0.], z, [zb]]))
                col_meta[(j, i)] = (M, h)
                cg1_map[j, i] = max(cg1_modal(N2v[:M, j, i], h), C_MIN)
        print(f'  {len(col_meta)} usable columns', flush=True)

    T = {k: np.full_like(U[k], np.nan) for k in U}
    KEYS = ('px', 'py', 'dx', 'dy', 'gx', 'gy')
    for (j, i), (M, h) in col_meta.items():
        hz = 0.5 * (h[:-1] + h[1:])
        c2_h = GAMMA * cg1_map[j, i] ** 2 / h
        cols = np.stack([U[k][:M, j, i] for k in KEYS], axis=1)
        tapd = fgnv_solve(np.nan_to_num(cols), N2v[:M, j, i] * hz, c2_h, GAMMA)
        for c, k in enumerate(KEYS):
            T[k][:M, j, i] = tapd[:, c]

    rx, ry = rhox.values, rhoy.values
    a_d = T['dx'] * rx + T['dy'] * ry
    a_p = T['px'] * rx + T['py'] * ry
    a_g = T['gx'] * rx + T['gy'] * ry            # unit-kappa GM release; scaled by kappa* at write
    mm = m2[None, :, :]
    apeint_d = np.nansum(np.where(mm, a_d, np.nan) * DZL[:, None, None], axis=0)
    apeint_p = np.nansum(np.where(mm, a_p, np.nan) * DZL[:, None, None], axis=0)
    apeint_g = np.nansum(np.where(mm, a_g, np.nan) * DZL[:, None, None], axis=0)
    ACC['ape_d'] += np.nan_to_num(apeint_d)
    ACC['ape_p'] += np.nan_to_num(apeint_p)
    ACC['ape_g'] += np.nan_to_num(apeint_g)
    ACC['int_d'] += np.nansum(apeint_d * np.where(m2, area, 0.))
    ACC['int_p'] += np.nansum(apeint_p * np.where(m2, area, 0.))
    ACC['int_g'] += np.nansum(apeint_g * np.where(m2, area, 0.))
    okz = np.isfinite(a_d) & np.isfinite(a_p) & np.isfinite(a_g) & mm
    # kappa* fit sums: volume-weighted LS of the diagnosed code flux onto +grad_h rho
    vol = np.where(okz, area[None, :, :] * DZL[:, None, None], 0.)
    Fxv, Fyv = one.Fx.astype('float64').values, one.Fy.astype('float64').values
    ACC['fit_fg'] += np.nansum(vol * (Fxv * rx + Fyv * ry))
    ACC['fit_gg'] += np.nansum(vol * (rx ** 2 + ry ** 2))
    ACC['sa_d'] += np.where(okz, a_d, 0.).sum(axis=2)
    ACC['sa_p'] += np.where(okz, a_p, 0.).sum(axis=2)
    ACC['sa_g'] += np.where(okz, a_g, 0.).sum(axis=2)
    ACC['su_d'] += np.where(okz, T['dy'], 0.).sum(axis=2)
    ACC['su_p'] += np.where(okz, T['py'], 0.).sum(axis=2)
    ACC['su_g'] += np.where(okz, T['gy'], 0.).sum(axis=2)
    ACC['sn'] += okz.sum(axis=2)
    nt += 1
    print(f'  [{nt}/{len(snaps)}] {os.path.basename(fn)} it={it} '
          f'running ratio {ACC["int_p"] / ACC["int_d"]:.3f}', flush=True)

# ----- write ------------------------------------------------------------------------------------
ny, nx = ACC['ape_d'].shape
cnt = np.maximum(ACC['sn'], 1)
kap = ACC['fit_fg'] / ACC['fit_gg']
out = xr.Dataset(
    dict(ape_diag=(('yh', 'xh'), ACC['ape_d'] / nt),
         ape_pred=(('yh', 'xh'), ACC['ape_p'] / nt),
         ape_gm=(('yh', 'xh'), kap * ACC['ape_g'] / nt),
         sect_a_diag=(('zl', 'yh'), ACC['sa_d'] / cnt),
         sect_a_pred=(('zl', 'yh'), ACC['sa_p'] / cnt),
         sect_a_gm=(('zl', 'yh'), kap * ACC['sa_g'] / cnt),
         sect_uy_diag=(('zl', 'yh'), ACC['su_d'] / cnt),
         sect_uy_pred=(('zl', 'yh'), ACC['su_p'] / cnt),
         sect_uy_gm=(('zl', 'yh'), kap * ACC['su_g'] / cnt),
         mask2d=(('yh', 'xh'), m2.astype('i1'))),
    coords=dict(zl=zl))
out.attrs.update(run=RUN, model=MODEL, factor=FACTOR, FGR=FGR, nt=nt,
                 gamma=GAMMA, c_min=C_MIN, rho0=RHO0,
                 kappa_star=float(kap),
                 ratio_pred_over_diag=float(ACC['int_p'] / ACC['int_d']),
                 ratio_gm_over_diag=float(kap * ACC['int_g'] / ACC['int_d']),
                 note='a_tap>0 = APE release; sponge+wall rim eroded 2 cells; seamount unmasked; '
                      'GM = constant kappa_star, volume-weighted flux LS fit (same loss as ANN)')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_netcdf(OUT)
print(f'\n=== channel woc_p0625 factor-{FACTOR}, {nt} snapshots ===')
print(f'  deployed APE release ratio pred/diag = {ACC["int_p"] / ACC["int_d"]:.3f}')
print(f'  fitted kappa* = {kap:.1f} m2/s (online sweep bracket: 500-2000)')
print(f'  deployed APE release ratio GM(kappa*)/diag = {kap * ACC["int_g"] / ACC["int_d"]:.3f}')
print(f'  wrote {OUT}')

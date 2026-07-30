"""The mean-flow forcing G under the deployed FGNV operator, completing what diag_fgnv_taper.py
did for the APE release.

v1 of this script used centred differences and np.gradient for the derivatives and FAILED its own
cross-check (clamp15 ratio 1.833 against the mean-impact reference 0.679 at factor-9): the
area-mean of depth-averaged G is a small residual of large cancellations, so it is fragile to the
discrete operators, and v1 also omitted the gradient floor so clipped +-15 values from effectively
unstratified points entered the budget. This version uses the audited xgcm flux-form operators from
SGS_skill_rho and replicates its clamp15 G exactly as the validation row.

Forcing in flux form (MOM6-faithful):  the skew flux of a transport Ups is
  F_skew_h = -Ups * rhoz ,   F_skew_z = a = Ups . grad_h(rho)
so G = -div_h(F_skew_h) - d/dz(a) = div_h(Ups*rhoz) - d/dz(a).  For the clamp15 validation row we
instead reproduce SGS_skill_rho verbatim: G = -div_h(F_clamped) - d/dz(a) with the flux clamp
(1e2), gradient floor (1e-10) and Upsilon clamp (15).

The deployed row tapers every usable column (no subsampling -- horizontal derivatives need
neighbours); cg1 (the per-column modal eigenproblem) is cached from the first snapshot, since
monthly stratification changes barely move the smoothing scale. Solver + cg1 from
diag_fgnv_taper.py, self-tested there.

  FACTOR=9  NSNAP=6
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
from scipy.linalg import solve_banded, eigh_tridiagonal
from helpers.cm26 import read_datasets, create_grid

FAC = int(os.environ.get('FACTOR', 9))
NSNAP = int(os.environ.get('NSNAP', 6))
GAMMA, C_MIN = 1.0, 0.01
N2_FLOOR = (1e-15 * 7.2921e-5) ** 2
FLUX_CLAMP, UPS_CLAMP, GRAD_FLOOR = 1.0e2, 15.0, 1.0e-10
PRED = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/predictions')


def fgnv_solve(psi, hN2, c2_h, gamma):
    M = len(hN2)
    ab = np.zeros((3, M))
    ab[1, :] = hN2 + c2_h[:M] + c2_h[1:]
    ab[0, 1:] = -c2_h[1:M]
    ab[2, :-1] = -c2_h[1:M]
    rhs = (1.0 + gamma) * (hN2[:, None] if psi.ndim > 1 else hN2) * psi
    return solve_banded((1, 1), ab, rhs)


def cg1_modal(N2, h):
    hz = 0.5 * (h[:-1] + h[1:])
    B = np.maximum(N2, 1e-14) * hz
    d = (1. / h[:-1] + 1. / h[1:]) / B
    e = -1. / (h[1:-1] * np.sqrt(B[:-1] * B[1:]))
    try:
        lam = eigh_tridiagonal(d, e, select='i', select_range=(0, 0), eigvals_only=True)[0]
    except Exception:
        lam = np.linalg.eigvalsh(np.diag(d) + np.diag(e, 1) + np.diag(e, -1))[0]
    return 1.0 / np.sqrt(max(lam, 1e-30))


ds = read_datasets(['test'], [FAC], subfilter='subfilter-neutral', FGR=3)[f'test-{FAC}']
pm = ds.param
grid = create_grid(pm)
zl = ds.data.zl.values
area = np.asarray(pm.dxT * pm.dyT)


def divh_flux(fx, fy):
    """The audited C-grid flux-form divergence (cm26.py SGS_skill_rho)."""
    return (grid.diff(grid.interp(fx.fillna(0.), 'X') * pm.dyCu, 'X')
            + grid.diff(grid.interp(fy.fillna(0.), 'Y') * pm.dxCv, 'Y')) / (pm.dxT * pm.dyT)


dzf = lambda f: -f.differentiate('zl')      # zl positive down

cg1_map, col_meta = None, None
# The area-mean-of-depth-avg statistic is dominated by the vertical boundary term a(surface)/H,
# which the FGNV taper removes BY CONSTRUCTION -- under the deployed operator that statistic
# collapses to near-zero noise (first run of this version: ratio -0.7 on ~0/~0). So the meaningful
# comparison is pattern-based: area-weighted regression slope and correlation of the depth-averaged
# maps, accumulated as second moments. The mean ratio is retained for the c15 row only, as the
# cross-check against the mean-impact reference.
S = {k: dict(t=0., p=0., dd=0., pp=0., pd=0.) for k in ['c15', 'dep']}


def accumulate(key, Gd, Gp):
    gd = Gd.mean('zl', skipna=True).values
    gp = Gp.mean('zl', skipna=True).values
    ok = np.isfinite(gd) & np.isfinite(gp)
    S[key]['t'] += np.sum(gd[ok] * area[ok]); S[key]['p'] += np.sum(gp[ok] * area[ok])
    S[key]['dd'] += np.sum(gd[ok] ** 2 * area[ok]); S[key]['pp'] += np.sum(gp[ok] ** 2 * area[ok])
    S[key]['pd'] += np.sum(gd[ok] * gp[ok] * area[ok])

for it in range(NSNAP):
    pfn = f'{PRED}/factor-{FAC}/test-{it:03d}.nc'
    if not os.path.exists(pfn):
        continue
    one = ds.data[['Fx', 'Fy', 'rhox', 'rhoy', 'N_buoyancy']].isel(time=it).load()
    st = xr.open_dataset(pfn)
    Pmap = {'x': st.Fx_pred.astype('float64'), 'y': st.Fy_pred.astype('float64')}
    st.close()
    Dmap = {'x': one.Fx.astype('float64'), 'y': one.Fy.astype('float64')}
    rhox, rhoy = one.rhox.astype('float64'), one.rhoy.astype('float64')
    N2 = (one.N_buoyancy.astype('float64') ** 2).clip(min=N2_FLOOR)
    rhoz = -(1025.0 / 9.8) * N2
    magx, magy = np.sqrt(rhox ** 2 + rhoz ** 2), np.sqrt(rhoy ** 2 + rhoz ** 2)

    # --- clamp15 validation row: SGS_skill_rho verbatim -----------------------------------------
    Gc = {}
    for tag, F_ in [('t', Dmap), ('p', Pmap)]:
        fx = F_['x'].clip(-FLUX_CLAMP, FLUX_CLAMP)
        fy = F_['y'].clip(-FLUX_CLAMP, FLUX_CLAMP)
        Ux = xr.where(magx < GRAD_FLOOR, 0., (fx / (magx + 1e-30)).clip(-UPS_CLAMP, UPS_CLAMP))
        Uy = xr.where(magy < GRAD_FLOOR, 0., (fy / (magy + 1e-30)).clip(-UPS_CLAMP, UPS_CLAMP))
        a = Ux * rhox + Uy * rhoy
        Gc[tag] = -divh_flux(fx, fy) - dzf(a)
    accumulate('c15', Gc['t'], Gc['p'])

    # --- deployed row: unclamped Upsilon through the FGNV solve ---------------------------------
    U = {'px': (Pmap['x'] / (magx + 1e-300)).values, 'py': (Pmap['y'] / (magy + 1e-300)).values,
         'dx': (Dmap['x'] / (magx + 1e-300)).values, 'dy': (Dmap['y'] / (magy + 1e-300)).values}
    N2v = N2.values
    ny, nx = N2v.shape[1], N2v.shape[2]
    if cg1_map is None:
        print(f'factor-{FAC}: cg1 + column metadata ({ny * nx} columns)...', flush=True)
        cg1_map = np.full((ny, nx), np.nan)
        col_meta = {}
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
    for (j, i), (M, h) in col_meta.items():
        hz = 0.5 * (h[:-1] + h[1:])
        c2_h = GAMMA * cg1_map[j, i] ** 2 / h
        cols = np.stack([U[k][:M, j, i] for k in ('px', 'py', 'dx', 'dy')], axis=1)
        tapd = fgnv_solve(np.nan_to_num(cols), N2v[:M, j, i] * hz, c2_h, GAMMA)
        for c, k in enumerate(('px', 'py', 'dx', 'dy')):
            T[k][:M, j, i] = tapd[:, c]

    wrap = lambda a_: one.Fx.astype('float64').copy(data=a_)
    Gt = {}
    for tag, kx, ky in [('t', 'dx', 'dy'), ('p', 'px', 'py')]:
        Uxt, Uyt = wrap(T[kx]), wrap(T[ky])
        a_t = Uxt * rhox + Uyt * rhoy
        Gt[tag] = -divh_flux(-(Uxt * rhoz), -(Uyt * rhoz)) - dzf(a_t)
    accumulate('dep', Gt['t'], Gt['p'])
    print(f'  snapshot {it} done', flush=True)

print(f'\nfactor-{FAC}, {NSNAP} snapshots (depth-avg G maps, area-weighted):')
for key, lab in [('c15', 'clamp15 (SGS-verbatim)'), ('dep', 'DEPLOYED (FGNV)     ')]:
    beta = S[key]['pd'] / S[key]['dd']
    corr = S[key]['pd'] / np.sqrt(S[key]['dd'] * S[key]['pp'])
    amp = np.sqrt(S[key]['pp'] / S[key]['dd'])
    print(f'  {lab}: slope {beta:6.3f}  corr {corr:6.3f}  amp {amp:6.3f}  '
          f'mean-ratio {S[key]["p"] / S[key]["t"]:7.3f}')
print('  [c15 mean-ratio cross-check refs: 0.844/0.679/0.609/0.568 at f4/9/12/15;'
      ' dep mean-ratio is boundary-term-collapsed noise, listed only for the record]')

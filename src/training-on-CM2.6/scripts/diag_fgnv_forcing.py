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
import gsw
import xarray as xr
from scipy.linalg import solve_banded, eigh_tridiagonal
from helpers.cm26 import read_datasets, create_grid, propagate_mask

FAC = int(os.environ.get('FACTOR', 9))
NSNAP = int(os.environ.get('NSNAP', 6))
GAMMA, C_MIN = 1.0, 0.01
N2_FLOOR = (1e-15 * 7.2921e-5) ** 2
FLUX_CLAMP, UPS_CLAMP, GRAD_FLOOR = 1.0e2, 15.0, 1.0e-10
PRED = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/predictions')
MAPS_OUT = os.environ.get('MAPS_OUT', '')   # if set, save the time-mean component maps here
# BELOW_ML=1: take every depth average only BELOW the mixed layer (de Boyer Montegut 0.03 in
# sigma0, computed from salt/temp -- the stored rho is locally referenced and would give a
# spuriously shallow ML). The deployed operator still acts on the FULL column; only the diagnostic
# averaging is restricted. This swaps the leaked boundary term from a(surface) to a(MLD).
BELOW_ML = os.environ.get('BELOW_ML', '0') == '1'
# SECT_OUT: save per-LEVEL second moments (area-weighted, across snapshots) of the forcing and its
# parts, for the vertical-section figure: at what depths is the deployed G a faithful forcing, and
# at what depths does the vertical term dominate in MAGNITUDE even where the divergence part wins
# the pattern correlation (the two can disagree).
SECT_OUT = os.environ.get('SECT_OUT', '')
# SECT_MASK=1: restrict the per-level stats to points >=2 cells from topography at that level
# (the audited away-metric convention, which these stats otherwise lack -- fillna(0) puts a
# divergence ring around every ridge) and >=2 levels above the local seafloor (the deepest wet
# level's dz term is NaN-edged and near-bottom Upsilon carries the abyssal weak-stratification
# pathology). Adjudicates whether the deep rise of the vertical part is physics or bottom handling.
SECT_MASK = os.environ.get('SECT_MASK', '0') == '1'
BELOWMASK = None


def zmean(F):
    return (F.where(BELOWMASK) if (BELOW_ML and BELOWMASK is not None) else F).mean('zl', skipna=True)


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
ny0, nx0 = pm.dxT.shape
NZ = len(zl)
DZL = np.gradient(zl)
PROF = {}
MASK3 = None
if SECT_MASK:
    wet3 = pm.wet if 'zl' in pm.wet.dims else pm.wet.expand_dims(zl=NZ)
    away = np.stack([np.asarray(propagate_mask(wet3.isel(zl=k), grid, niter=2)) > 0.5
                     for k in range(NZ)])
    kb = np.asarray(wet3.sum('zl')).astype(int)          # wet levels per column (contiguous)
    lev = np.arange(NZ)[:, None, None]
    above_bot = lev < np.maximum(kb - 2, 0)[None]        # drop the two deepest wet levels
    MASK3 = away & above_bot
    print(f'SECT_MASK on: keeping {MASK3.sum()/max(np.asarray(wet3.values).sum(),1):.2f} '
          f'of wet cells', flush=True)


def lvl_acc(name, X, Y=None):
    # area-weighted per-level sums of X*Y (X^2 if Y is None), plus the weights
    Xv = X.values if hasattr(X, 'values') else X
    Yv = Xv if Y is None else (Y.values if hasattr(Y, 'values') else Y)
    ok = np.isfinite(Xv) & np.isfinite(Yv)
    if MASK3 is not None:
        ok = ok & MASK3
    if name not in PROF:
        PROF[name] = np.zeros(NZ); PROF[name + '_w'] = np.zeros(NZ)
    PROF[name] += np.sum(np.where(ok, Xv * Yv, 0.) * area[None], axis=(1, 2))
    PROF[name + '_w'] += np.sum(ok * area[None], axis=(1, 2))

MAPS = {k: np.zeros((ny0, nx0)) for k in
        ['c15_total', 'c15_total_p', 'c15_vert', 'c15_div', 'dep_diag', 'dep_pred',
         'dep_band_diag', 'dep_band_pred', 'dep_band_n', 'dep_ape_diag', 'dep_ape_pred']}
NMAP = 0
# The area-mean-of-depth-avg statistic is dominated by the vertical boundary term a(surface)/H,
# which the FGNV taper removes BY CONSTRUCTION -- under the deployed operator that statistic
# collapses to near-zero noise (first run of this version: ratio -0.7 on ~0/~0). So the meaningful
# comparison is pattern-based: area-weighted regression slope and correlation of the depth-averaged
# maps, accumulated as second moments. The mean ratio is retained for the c15 row only, as the
# cross-check against the mean-impact reference.
S = {k: dict(t=0., p=0., dd=0., pp=0., pd=0.) for k in ['c15', 'dep']}


def accumulate(key, Gd, Gp):
    gd = zmean(Gd).values
    gp = zmean(Gp).values
    ok = np.isfinite(gd) & np.isfinite(gp)
    S[key]['t'] += np.sum(gd[ok] * area[ok]); S[key]['p'] += np.sum(gp[ok] * area[ok])
    S[key]['dd'] += np.sum(gd[ok] ** 2 * area[ok]); S[key]['pp'] += np.sum(gp[ok] ** 2 * area[ok])
    S[key]['pd'] += np.sum(gd[ok] * gp[ok] * area[ok])

for it in range(NSNAP):
    pfn = f'{PRED}/factor-{FAC}/test-{it:03d}.nc'
    if not os.path.exists(pfn):
        continue
    one = ds.data[['Fx', 'Fy', 'rhox', 'rhoy', 'N_buoyancy', 'salt', 'temp']].isel(time=it).load()
    if BELOW_ML:
        sig = 1000. + gsw.sigma0(one.salt, one.temp)
        exc = (sig - sig.isel(zl=0)) > 0.03
        mld = xr.where(exc.any('zl'), exc.idxmax('zl'), float(one.zl.max()))
        BELOWMASK = one.zl > mld
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
        if MAPS_OUT and tag == 't':
            MAPS['c15_vert'] += np.nan_to_num(zmean(-dzf(a)).values)
            MAPS['c15_div'] += np.nan_to_num(zmean(-divh_flux(fx, fy)).values)
        if SECT_OUT and tag == 't':
            Vt, Dt = -dzf(a), -divh_flux(fx, fy)
            lvl_acc('gt2', Gc['t']); lvl_acc('vt2', Vt); lvl_acc('dt2', Dt)
            lvl_acc('gtvt', Gc['t'], Vt); lvl_acc('gtdt', Gc['t'], Dt)
    accumulate('c15', Gc['t'], Gc['p'])
    if MAPS_OUT:
        MAPS['c15_total'] += np.nan_to_num(zmean(Gc['t']).values)
        MAPS['c15_total_p'] += np.nan_to_num(zmean(Gc['p']).values)

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
        if MAPS_OUT:
            # depth-INTEGRATED deployed APE release, sum(a_tap*dz): with Upsilon tapered, the
            # release is interior by construction -- no mixed-layer cut needed, the operator
            # does the separation. MASK3 honoured for consistency with the band forcing maps.
            # NO topography/bottom mask here: a_tap is a pointwise product (no horizontal
            # derivative, hence no edge-ring artifact to guard against), and masking an energy
            # budget deletes the near-topography release hotspots and biases the ratio (first
            # masked run: 1.26 vs ~1.09 unmasked). The mask stays on the FORCING maps, which
            # genuinely carry a divergence.
            Xa = a_t.values
            MAPS['dep_ape_diag' if tag == 't' else 'dep_ape_pred'] += \
                np.nan_to_num(np.nansum(Xa * DZL[:, None, None], axis=0))
    accumulate('dep', Gt['t'], Gt['p'])
    if MAPS_OUT:
        MAPS['dep_diag'] += np.nan_to_num(zmean(Gt['t']).values)
        MAPS['dep_pred'] += np.nan_to_num(zmean(Gt['p']).values)
        # interior-band (300-3000 m) deployed-forcing maps for the main-text figure; the shared
        # finite-mask of the diagnosed field is applied to both so the two maps are comparable,
        # and MASK3 (topography/bottom exclusion) is honoured when SECT_MASK is on
        bnd = (zl >= 300.) & (zl <= 3000.)
        Xd, Xp = Gt['t'].values[bnd], Gt['p'].values[bnd]
        if MASK3 is not None:
            m3 = MASK3[bnd]
            Xd, Xp = np.where(m3, Xd, np.nan), np.where(m3, Xp, np.nan)
        okb = np.isfinite(Xd) & np.isfinite(Xp)
        MAPS['dep_band_diag'] += np.nansum(np.where(okb, Xd, 0.), axis=0)
        MAPS['dep_band_pred'] += np.nansum(np.where(okb, Xp, 0.), axis=0)
        MAPS['dep_band_n'] += okb.sum(axis=0)
    if SECT_OUT:
        lvl_acc('gd2', Gt['t']); lvl_acc('gp2', Gt['p'])
        lvl_acc('gtgd', Gc['t'], Gt['t']); lvl_acc('gdgp', Gt['t'], Gt['p'])
    NMAP += 1
    print(f'  snapshot {it} done', flush=True)

if SECT_OUT:
    out = xr.Dataset({k: (('zl',), v) for k, v in PROF.items()}, coords={'zl': zl})
    out.attrs['n_snapshots'] = NMAP
    out.to_netcdf(SECT_OUT)
    print(f'wrote {SECT_OUT}', flush=True)

if MAPS_OUT:
    wet0 = np.isfinite(np.asarray(ds.data.Fx.isel(time=0, zl=0).values))
    out = xr.Dataset({k: (('yh', 'xh'), np.where(wet0, v / max(NMAP, 1), np.nan))
                      for k, v in MAPS.items()})
    out.attrs['n_snapshots'] = NMAP
    out.to_netcdf(MAPS_OUT)
    print(f'wrote {MAPS_OUT}', flush=True)

print(f'\nfactor-{FAC}, {NSNAP} snapshots (depth-avg G maps, area-weighted):')
for key, lab in [('c15', 'clamp15 (SGS-verbatim)'), ('dep', 'DEPLOYED (FGNV)     ')]:
    beta = S[key]['pd'] / S[key]['dd']
    corr = S[key]['pd'] / np.sqrt(S[key]['dd'] * S[key]['pp'])
    amp = np.sqrt(S[key]['pp'] / S[key]['dd'])
    print(f'  {lab}: slope {beta:6.3f}  corr {corr:6.3f}  amp {amp:6.3f}  '
          f'mean-ratio {S[key]["p"] / S[key]["t"]:7.3f}')
print('  [c15 mean-ratio cross-check refs: 0.844/0.679/0.609/0.568 at f4/9/12/15;'
      ' dep mean-ratio is boundary-term-collapsed noise, listed only for the record]')

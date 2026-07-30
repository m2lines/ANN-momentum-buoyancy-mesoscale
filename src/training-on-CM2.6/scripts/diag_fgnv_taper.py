"""How Upsilon is modified online, reproduced offline with MOM6's actual operator chain.

The deployed chain (verified against MOM_parameter_doc.all of the canonical 1/4-deg channel run
tau_0.2_cb_1.0_cu_0.0_neutral, and against the Fortran):

  1. Upsilon from the ANN flux, UNCLAMPED -- the canonical binary has no clamp parameters at all
     (no MESO_UPSILON_CLAMP / FLUX_CLAMP / MAG_GRAD_FLOOR in its parameter doc).
  2. MESO_SFN_MIN_DIST_BOUNDARY = 50 m boundary zeroing: **gated on `.not. use_EOS`**
     (MOM_meso_sfn_ANN.F90:418, "In layered mode, skip interfaces at the bottom or surface").
     The channel runs USE_EOS=True / EQN_OF_STATE=LINEAR, so it does NOT fire there -- set
     MIN_DIST=0 (the default here) for the channel/CM2.6 chain. It DOES fire in stacked-layer
     configs (NeverWorld2), where use_EOS is false: set MIN_DIST=50 to model those.
  3. Added to the GM streamfunction (zero here: KHTH = 0), bottom dense-water guard (skipped, a
     boundary-cell detail).
  4. KHTH_USE_FGNV_STREAMFUNCTION = True: scale by (1 + FGNV_FILTER_SCALE), then solve
        N^2 psi - d/dz( c^2 dpsi/dz ) = N^2 (1+gamma) psi_unlim ,  psi = 0 at surface and bottom
     with c^2 = gamma * cg1^2 (MOM_thickness_diffuse.F90:1160-1180, streamfn_solver:1761).
     Constant-N transfer function: T(m) = (1+gamma)/(1+gamma (cg1 m/N)^2) -- unit gain at mode 1,
     (1+gamma) for broader structure, low-pass above.
     gamma = FGNV_FILTER_SCALE = 1.0 ; cg1 floored at FGNV_C_MIN = 0.01 m/s ;
     N2 floored at (FGNV_STRAT_FLOOR*Omega)^2 = (1e-15*7.29e-5)^2 ~ 5e-39 s-2 (negligible; kept).
  5. Downstream transport limiters (Sfn_safe / CFL-type) -- NOT replicated; they cap extreme
     transports against available water, noted where relevant.

cg1 is the first-baroclinic wave speed. MOM6 solves the vertical modal eigenproblem
(MOM_wave_speed); we do the same per column -- w'' = -(N^2/c^2) w, w = 0 at surface and local
bottom -- via a symmetric tridiagonal eigenvalue solve, with the WKB estimate (1/pi) int N dz
printed for comparison.

Differences from the deployed code that remain: everything at tracer columns (not u/v faces),
N_buoyancy at the 50 zl levels treated as interface values with np.gradient thicknesses, and the
step-3/5 guards skipped.

v1 of this script (superseded) solved the full 50-level column everywhere -- below the seafloor the
operator degenerates to pure smoothing, wrecking the bottom pinning on shelves -- used WKB cg1,
tapered |Upsilon| instead of the components, applied a 15 m2/s clamp the deployed binary does not
have, and knew nothing of the 50 m boundary zeroing. Its tridiagonal port itself is kept below as
_fgnv_column_v1 and verified against an independent solver in the self-tests.

  FACTOR=9  NSNAP=2  STEP=3
"""
import os, sys
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded, eigh_tridiagonal
from helpers.cm26 import read_datasets

FAC = int(os.environ.get('FACTOR', 9))
NSNAP = int(os.environ.get('NSNAP', 2))
STEP = int(os.environ.get('STEP', 3))
GAMMA, C_MIN = 1.0, 0.01
MIN_DIST = float(os.environ.get('MIN_DIST', 0.0))
N2_FLOOR = (1e-15 * 7.2921e-5) ** 2
PRED = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/predictions')
FGNV_SCALE = GAMMA   # name used by the v1 reference implementation below


# ----- v1 reference implementation, verbatim (self-test target) ---------------------------------
def _dz_iface_v1(dz):
    out = np.zeros(len(dz) + 1)
    out[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    out[0], out[-1] = dz[0], dz[-1]
    return out


def _fgnv_column_v1(psi_in, N2, dz, c2):
    nk = len(dz)
    sfn = np.concatenate([[0.], psi_in[1:nk], [0.]]) * (1.0 + FGNV_SCALE)
    hN2 = N2 * _dz_iface_v1(dz)
    c2_h = c2 / np.maximum(dz, 1e-9)
    c1 = np.zeros(nk + 1)
    b_denom = hN2[1] + c2_h[0]
    beta = 1.0 / (b_denom + c2_h[1])
    d1 = beta * b_denom
    sfn[1] = beta * hN2[1] * sfn[1]
    for k in range(2, nk):
        c1[k - 1] = beta * c2_h[k - 1]
        b_denom = hN2[k] + d1 * c2_h[k - 1]
        beta = 1.0 / (b_denom + c2_h[k])
        d1 = beta * b_denom
        sfn[k] = beta * (hN2[k] * sfn[k] + c2_h[k - 1] * sfn[k - 1])
    c1[nk - 1] = beta * c2_h[nk - 1]
    sfn[nk] = 0.
    for k in range(nk - 1, 0, -1):
        sfn[k] = sfn[k] + c1[k] * sfn[k + 1]
    return sfn
# ------------------------------------------------------------------------------------------------


def fgnv_solve(psi, hN2, c2_h, gamma):
    """Independent solver for the same tridiagonal system, via scipy banded LU.
    psi: (M,) or (M,nrhs) interior values; hN2: (M,); c2_h: (M+1,) = c^2/h per layer."""
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


# ----- self-tests -------------------------------------------------------------------------------
rng = np.random.default_rng(0)
nk = 50
dz_t = rng.uniform(5., 300., nk)
N2_t = rng.uniform(1e-7, 1e-4, nk + 1)
c2_t = np.full(nk, 4.0)
u_t = rng.normal(size=nk)
v1 = _fgnv_column_v1(np.concatenate([[0.], u_t]), N2_t, dz_t, c2_t)
hN2_t = (N2_t * _dz_iface_v1(dz_t))[1:nk]
new = fgnv_solve(u_t[:nk - 1], hN2_t, c2_t / np.maximum(dz_t, 1e-9), GAMMA)
err = np.max(np.abs(v1[1:nk] - new)) / np.max(np.abs(new))
print(f'TEST solver port vs independent banded LU : rel err {err:.2e}  '
      f'({"PASS" if err < 1e-10 else "FAIL"})')

idn = fgnv_solve(u_t[:nk - 1], hN2_t, 0. * c2_t[:nk], 0.0)
err0 = np.max(np.abs(idn - u_t[:nk - 1]))
print(f'TEST gamma=0, c=0 -> identity             : max err {err0:.2e}  '
      f'({"PASS" if err0 < 1e-12 else "FAIL"})')

H, M, N0 = 4000., 199, 2e-3
h_u = np.full(M + 1, H / (M + 1))
zc = np.cumsum(h_u)[:M]
cg = cg1_modal(np.full(M, N0 ** 2), h_u)
print(f'TEST modal cg1, const N                   : {cg:.3f} vs NH/pi = {N0 * H / np.pi:.3f}  '
      f'({"PASS" if abs(cg / (N0 * H / np.pi) - 1) < 0.01 else "FAIL"})')

mode1 = np.sin(np.pi * zc / H)
hN2_u = N0 ** 2 * h_u[0] * np.ones(M)
tap1 = fgnv_solve(mode1, hN2_u, GAMMA * cg ** 2 / h_u, GAMMA)
gain = np.sum(tap1 * mode1) / np.sum(mode1 * mode1)
print(f'TEST mode-1 gain (should be ~1)           : {gain:.3f}  '
      f'({"PASS" if abs(gain - 1) < 0.03 else "FAIL"})')
mode8 = np.sin(8 * np.pi * zc / H)
tap8 = fgnv_solve(mode8, hN2_u, GAMMA * cg ** 2 / h_u, GAMMA)
g8 = np.sum(tap8 * mode8) / np.sum(mode8 * mode8)
print(f'TEST mode-8 gain (low-pass, ~2/65)        : {g8:.3f}')

# ----- data -------------------------------------------------------------------------------------
ds = read_datasets(['test'], [FAC], subfilter='subfilter-neutral', FGR=3)[f'test-{FAC}']
zl = ds.data.zl.values
nz = len(zl)
acc = {k: np.zeros(nz) for k in ['raw', 'zero50', 'tap', 'n', 'nphys']}
S = dict(a_raw_p=0., a_raw_d=0., a_dep_p=0., a_dep_d=0., a_c15_p=0., a_c15_d=0.)
cgm, cgw = [], []
tail = dict(mx=0., gt15=0, npt=0, spike=0)
ncol = nshelf = 0

for it in range(NSNAP):
    pfn = f'{PRED}/factor-{FAC}/test-{it:03d}.nc'
    if not os.path.exists(pfn):
        continue
    one = ds.data.isel(time=it)
    st = xr.open_dataset(pfn)
    Px, Py = np.asarray(st.Fx_pred.values, 'float64'), np.asarray(st.Fy_pred.values, 'float64')
    st.close()
    Fx, Fy = np.asarray(one.Fx.values, 'float64'), np.asarray(one.Fy.values, 'float64')
    rhox, rhoy = np.asarray(one.rhox.values, 'float64'), np.asarray(one.rhoy.values, 'float64')
    N2 = np.maximum(np.asarray(one.N_buoyancy.values, 'float64') ** 2, N2_FLOOR)
    rhoz = -(1025.0 / 9.8) * N2
    magx, magy = np.sqrt(rhox ** 2 + rhoz ** 2), np.sqrt(rhoy ** 2 + rhoz ** 2)
    # deployed: no clamps at all (division guarded only against exact zero)
    Uxp, Uyp = Px / (magx + 1e-300), Py / (magy + 1e-300)
    Uxd, Uyd = Fx / (magx + 1e-300), Fy / (magy + 1e-300)

    ny, nx = Px.shape[1], Px.shape[2]
    for idx in range(0, ny * nx, STEP):
        j, i = divmod(idx, nx)
        w = np.isfinite(Uxp[:, j, i]) & np.isfinite(N2[:, j, i])
        M = int(w.sum())
        if M < 8 or not w[:M].all():        # require a contiguous wet column from the surface
            continue
        z = zl[:M]
        zb = z[-1] + 0.5 * (z[-1] - z[-2])
        h = np.diff(np.concatenate([[0.], z, [zb]]))
        n2 = N2[:M, j, i]
        hz = 0.5 * (h[:-1] + h[1:])
        cg_m = max(cg1_modal(n2, h), C_MIN)
        cgm.append(cg_m); cgw.append(np.sum(np.sqrt(n2) * hz) / np.pi)
        c2_h = GAMMA * cg_m ** 2 / h
        hN2 = n2 * hz

        cols = np.stack([Uxp[:M, j, i], Uyp[:M, j, i], Uxd[:M, j, i], Uyd[:M, j, i]], axis=1)
        near = (z < MIN_DIST) | ((zb - z) < MIN_DIST)      # step 2: 50 m boundary zeroing
        colz = np.where(near[:, None], 0., cols)
        tapd = fgnv_solve(colz, hN2, c2_h, GAMMA)

        up_raw = np.hypot(cols[:, 0], cols[:, 1])
        up_z50 = np.hypot(colz[:, 0], colz[:, 1])
        up_tap = np.hypot(tapd[:, 0], tapd[:, 1])
        # The unclamped division produces |Ups| up to ~1e33 where the gradient underflows; those
        # points would wreck a mean profile while the integrals (bounded by |F|) and the tapered
        # field (RHS weight is N^2, which vanishes exactly there) are immune. For the PROFILE
        # display only, accumulate the physical part |Ups| <= 100 m2/s (the CM2.6 99th percentile
        # is 45-73, SI Text S2) and count what is excluded; nothing else is masked.
        phys = up_raw <= 1e2
        acc['raw'][:M] += np.where(phys, up_raw, 0.); acc['zero50'][:M] += np.where(phys, up_z50, 0.)
        acc['tap'][:M] += up_tap; acc['n'][:M] += 1
        acc['nphys'][:M] += phys.astype(float)
        tail['spike'] += int((~phys).sum())

        rx, ry = rhox[:M, j, i], rhoy[:M, j, i]
        S['a_raw_p'] += np.sum((cols[:, 0] * rx + cols[:, 1] * ry) * hz)
        S['a_raw_d'] += np.sum((cols[:, 2] * rx + cols[:, 3] * ry) * hz)
        S['a_dep_p'] += np.sum((tapd[:, 0] * rx + tapd[:, 1] * ry) * hz)
        S['a_dep_d'] += np.sum((tapd[:, 2] * rx + tapd[:, 3] * ry) * hz)
        c15 = np.clip(cols, -15., 15.)                     # what SGS_skill_rho's clamp would do
        S['a_c15_p'] += np.sum((c15[:, 0] * rx + c15[:, 1] * ry) * hz)
        S['a_c15_d'] += np.sum((c15[:, 2] * rx + c15[:, 3] * ry) * hz)

        tail['mx'] = max(tail['mx'], up_raw.max()); tail['gt15'] += int((up_raw > 15).sum())
        tail['npt'] += M
        ncol += 1; nshelf += int(zb < 3000.)

n = np.maximum(acc['n'], 1)
np_ = np.maximum(acc['nphys'], 1)
raw, z50, tap = acc['raw'] / np_, acc['zero50'] / np_, acc['tap'] / n
print(f'\ncolumns {ncol} ({100 * nshelf / max(ncol, 1):.0f}% shallower than 3000 m), '
      f'{NSNAP} snapshots, factor-{FAC}')
print(f'cg1: modal median {np.median(cgm):.2f} m/s, WKB median {np.median(cgw):.2f} '
      f'(WKB/modal {np.median(np.array(cgw) / np.array(cgm)):.2f})')
print(f'unclamped tail: max|Ups| {tail["mx"]:.1e}, frac>15 m2/s {tail["gt15"] / tail["npt"]:.4f}, '
      f'frac>100 (excluded from profile display) {tail["spike"] / tail["npt"]:.4f}')
print(f'\n{"z (m)":>7} {"raw":>10} {"after zero":>10} {"tapered":>10} {"final/raw":>9}')
for zc_ in [5, 25, 55, 110, 180, 330, 525, 1000, 2000, 3500]:
    k = int(np.argmin(np.abs(zl - zc_)))
    r = tap[k] / raw[k] if raw[k] > 0 else np.nan
    print(f'{zl[k]:7.0f} {raw[k]:10.3e} {z50[k]:10.3e} {tap[k]:10.3e} {r:9.2f}')
print(f'\nAPE-sink integral (pred/diag):  raw unclamped {S["a_raw_p"] / S["a_raw_d"]:.3f}   '
      f'clamp15 {S["a_c15_p"] / S["a_c15_d"]:.3f}   DEPLOYED (MIN_DIST={MIN_DIST:.0f}m + FGNV) '
      f'{S["a_dep_p"] / S["a_dep_d"]:.3f}')
print(f'deployed/raw amplitude, diagnosed: {S["a_dep_d"] / S["a_raw_d"]:.2f}   '
      f'predicted: {S["a_dep_p"] / S["a_raw_p"]:.2f}')

fig, ax = plt.subplots(1, 2, figsize=(10.5, 5), constrained_layout=True)
ax[0].plot(raw, zl, 'k-', lw=2, label=r'raw $\Upsilon$ (unclamped)')
if MIN_DIST > 0:
    ax[0].plot(z50, zl, 'C0-', lw=1.4, label=f'after {MIN_DIST:.0f} m boundary zeroing')
ax[0].plot(tap, zl, 'C3--', lw=2, label='after FGNV solve (deployed)')
ax[0].set_yscale('log'); ax[0].set_ylim(zl.max(), zl.min()); ax[0].legend(fontsize=9)
ax[0].set_xlabel(r'mean $|\Upsilon|$ [m$^2$ s$^{-1}$]'); ax[0].set_ylabel('depth [m]')
ax[0].grid(alpha=0.3, which='both'); ax[0].set_xscale('log')
r_ = np.where(raw > 0, tap / raw, np.nan)
ax[1].plot(r_, zl, 'C0-', lw=2); ax[1].axvline(1.0, color='0.5', ls=':')
ax[1].axhline(MIN_DIST, color='0.7', lw=0.8)
ax[1].set_yscale('log'); ax[1].set_ylim(zl.max(), zl.min()); ax[1].set_xlim(0, 2.2)
ax[1].set_xlabel('deployed / raw'); ax[1].grid(alpha=0.3, which='both')
ax[1].set_title('net modification factor', fontsize=10)
fig.suptitle(f'Deployed modification of $\\Upsilon$: MIN\\_DIST={MIN_DIST:.0f} m + FGNV '
             f'($\\gamma$={GAMMA}, C\\_MIN={C_MIN}), factor-{FAC}', fontsize=11)
fig.savefig(f'fgnv_taper_f{FAC}.png', bbox_inches='tight', dpi=150)
print(f'wrote fgnv_taper_f{FAC}.png')

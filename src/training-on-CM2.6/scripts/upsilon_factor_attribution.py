"""Dhruv's question: the channel produces far more large-Upsilon points than CM2.6 -- is that the
ANN OUTPUT distribution differing, or the NORMALIZATION factors in front?

Exact factorization of what the code computes:
    Upsilon = |ANN_out| * (areaT * ||[grad u]_3x3||) * ( ||[grad rho]_3x3|| / |grad_3 rho| ) * C_ANN
              \_ network _/  \______ kappa_eff ______/   \______ 3 x slope ______/
so log Upsilon splits additively into three measurable pieces:
  * network      -- is the ANN behaving differently on channel inputs (out-of-distribution)?
  * kappa_eff    -- the dimensional prefactor (cell area x strain), i.e. "the normalization"
  * ratio        -- ||grad_h rho||_stencil / |grad_3 rho|, i.e. ~3x the effective isopycnal
                    slope; it is the piece that blows up when the LOCAL 3-D gradient collapses
The same hand-rolled forward pass is applied to both datasets so nothing differs but the data."""
import numpy as np, xarray as xr, torch, sys
sys.path.append('/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6')
from helpers.ann_tools import import_ANN

MODEL = '/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/model/ann_instance.nc'
B, RE, CB, SPONGE_LAT = '/scratch/db194/mom6/feb2026', 6.371e6, 1.0, -30.625
RHO0_T, G_T = 1025.0, 9.8

def stencil(a, n=3):
    s = n // 2; out = []
    for dj in range(-s, s + 1):
        for di in range(-s, s + 1):
            out.append(np.roll(np.take(a, np.clip(np.arange(a.shape[-2]) + dj, 0, a.shape[-2] - 1),
                                       axis=-2), -di, axis=-1))
    return np.stack(out, axis=-1)

ann = import_ANN(MODEL).double().eval()

def pieces(rhox, rhoy, sh_xx, sh_xy, vort, drdz, areaT, keep):
    """Return the three log-factors and Upsilon, on the common (z,y,x) grid."""
    S = [stencil(f) for f in (rhox, rhoy, sh_xx, sh_xy, vort)]
    rn = np.sqrt((S[0] ** 2 + S[1] ** 2).sum(-1))
    vn = np.sqrt((S[2] ** 2 + S[3] ** 2 + S[4] ** 2).sum(-1))
    ok = (rn > 0) & (vn > 0) & np.isfinite(rn) & np.isfinite(vn) & keep
    x = np.concatenate([S[0] / rn[..., None], S[1] / rn[..., None], S[2] / vn[..., None],
                        S[3] / vn[..., None], S[4] / vn[..., None]], axis=-1)
    with torch.no_grad():
        o = ann(torch.from_numpy(x[ok])).numpy()
    net = np.abs(o[:, 0])                                  # zonal component, nondimensional
    kap = (vn * np.broadcast_to(areaT, rn.shape))[ok] * CB  # m2/s
    mag = np.sqrt(rhox ** 2 + drdz ** 2)[ok]
    ratio = rn[ok] / mag                                    # ~ 3x the effective slope; blows up where local grad_3 rho collapses
    good = (net > 0) & (kap > 0) & (ratio > 0) & np.isfinite(ratio)
    return net[good], kap[good], ratio[good], (net * kap * ratio)[good]

rows = {}
for tag, dd in [("channel 1/4", 0.25), ("channel 1", 1.0)]:
    sub = {"channel 1/4": "p25", "channel 1": "1p0"}[tag]
    rd = f"{B}/channel_extra_sponge_slow_woc_{sub}/tau_0.2_cb_1.0_cu_0.0_neutral/output"
    ds = xr.open_dataset(f"{rd}/prog_z_010100.nc", decode_times=False).isel(Time=-1)
    rho = ds["rhopot2"].values.astype("f8")
    u, v = ds["u"].values.astype("f8"), ds["v"].values.astype("f8")
    lat, z = ds["yh"].values, ds["z_l"].values; ds.close()
    dy = np.deg2rad(dd) * RE; dx = dy * np.cos(np.deg2rad(lat))[None, :, None]
    drdx_f = np.diff(np.concatenate([rho[..., -1:], rho], axis=-1), axis=-1) / dx
    rhox = 0.5 * (drdx_f + np.roll(drdx_f, -1, axis=-1))
    rr = np.concatenate([rho[:, :1], rho, rho[:, -1:]], axis=-2)
    rhoy = 0.5 * (np.diff(rr, axis=-2)[:, :-1] + np.diff(rr, axis=-2)[:, 1:]) / dy
    uc = 0.5 * (u[..., :-1] + u[..., 1:]); vc = 0.5 * (v[:, :-1] + v[:, 1:])
    dudx = np.diff(u, axis=-1) / dx; dvdy = np.diff(v, axis=-2) / dy
    dudy = np.gradient(uc, axis=-2) / dy
    dvdx = (np.roll(vc, -1, -1) - np.roll(vc, 1, -1)) / (2 * dx)
    keep = np.isfinite(rho) & (np.broadcast_to(lat[None, :, None], rho.shape) < SPONGE_LAT)
    rows[tag] = pieces(rhox, rhoy, dudx - dvdy, dudy + dvdx, dvdx - dudy,
                       np.gradient(rho, z, axis=0), dx * dy, keep)

d = xr.open_dataset("/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-9/test-0.nc")
p = xr.open_dataset("/scratch/db194/CM26_datasets/ocean3d/subfilter-neutral/FGR3/factor-9/param.nc")
g = lambda k: d[k].values.astype("f8")
areaT = (p["dxT"].values * p["dyT"].values).astype("f8")[None]
drdz_t = -(RHO0_T / G_T) * g("N_buoyancy") ** 2
keep_t = np.isfinite(g("rhox")) & (p["wet"].values > 0.5)
rows["CM2.6 0.9deg"] = pieces(g("rhox"), g("rhoy"), g("sh_xx"), g("sh_xy_h"), g("rel_vort_h"),
                              drdz_t, areaT, keep_t)
d.close(); p.close()

print(f"{'dataset':<14}{'|ANN out|':>12}{'kappa_eff':>12}{'3 x slope':>15}{'Upsilon':>12}{'>15':>8}")
print(f"{'':14}{'(nondim)':>12}{'(m2/s)':>12}{'(3 x slope)':>15}{'(m2/s)':>12}")
for k, (net, kap, ratio, U) in rows.items():
    print(f"{k:<14}{np.median(net):>12.4f}{np.median(kap):>12.1f}{np.median(ratio):>15.4f}"
          f"{np.median(U):>12.3f}{100*np.mean(U > 15):>7.1f}%")
print(f"\n{'ratio to CM2.6':<14}{'|ANN out|':>12}{'kappa_eff':>12}{'3 x slope':>15}{'Upsilon':>12}")
b_net, b_kap, b_rat, b_U = [np.median(a) for a in rows["CM2.6 0.9deg"]]
for k in ["channel 1", "channel 1/4"]:
    net, kap, ratio, U = [np.median(a) for a in rows[k]]
    print(f"{k:<14}{net/b_net:>11.2f}x{kap/b_kap:>11.2f}x{ratio/b_rat:>14.2f}x{U/b_U:>11.2f}x")
print("\nTail comparison (p99 of each factor):")
for k, (net, kap, ratio, U) in rows.items():
    print(f"  {k:<14} net {np.percentile(net,99):8.3f}   kappa {np.percentile(kap,99):9.1f}"
          f"   ratio {np.percentile(ratio,99):12.1f}   Upsilon {np.percentile(U,99):10.1f}")

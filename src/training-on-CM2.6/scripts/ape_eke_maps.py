"""Pavel-Figure-2 analog for the channel (Part 2 sec 4 B-axis): depth-integrated dAPE and dEKE maps
at 1/4deg — does each closure change the coarse energy field in the same pattern as resolving eddies?
Rows: (filtered truth - no-param), (ANN - no-param), (MEKE - no-param). Cols: dAPE, dEKE [J/m2].

APE estimator (coordinate notes, agreed 2026-07-12): interface-displacement form in rho2 space,
APE(y,x) = sum_i 0.5 * g * drho_i * (e_i - R_i)^2  [J/m2], with e_i from cumsum of time-mean thkcello
on the rho2 diag grid (LINEAR EOS => g'_i = g*drho_i/rho0 exact). This is the MEAN APE (thkcello is a
time mean; quadratic in eta => eddy APE not included; EKE panel covers eddy energetics). Reference
R_i = truth's time+horizontal-mean interface depth per class — common across all runs; the reference
does NOT cancel in differences, so it is a stated convention. rho2 form avoids the z-form's 1/N^2
blowup in the weak abyssal stratification and handles outcrops (vanished classes pin to surface/bottom).
EKE maps: column sum of 0.5*h*(var_t u + var_t v) from native snapshots, rho0-weighted -> J/m2."""
import xarray as xr, numpy as np, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO0, G = 1035.0, 9.8
B = "/scratch/db194/mom6/feb2026"
P25 = f"{B}/channel_extra_sponge_slow_woc_p25"
TRUTH = f"{B}/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0"
RUNS = {  # 1/4deg closures vs no-param
    "no-param": (f"{P25}/tau_0.2_cb_0.0_cu_0.0", 4000),
    "ANN":      (f"{P25}/tau_0.2_cb_1.0_cu_0.0_neutral", 11500),   # canonical EXP_neutral_all4
    "MEKE":     (f"{P25}/tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500),
}
FAC = 4  # truth (1/16) -> 1/4deg block filter

def dl(da, *k):
    for d in da.dims:
        if any(x == d or d.startswith(x) for x in k): return d
    return None

def interfaces(rundir, cut):
    """time-mean interface depths e(rho2_i; y, x) from thkcello windows >= cut (positive down)."""
    fs = sorted(glob.glob(f"{rundir}/output/prog_rho_tmean_*.nc"))
    fs = [f for f in fs if int(f.split("_")[-1].split(".")[0]) >= cut]
    th = xr.open_mfdataset(fs, combine="by_coords", decode_times=False)["thkcello"]
    t = dl(th, "Time")
    th = th.mean(dim=t) if t else th
    th = th.fillna(0.0)
    rho = dl(th, "rho2", "rho")
    e = th.cumsum(dim=rho)            # depth of the BOTTOM interface of each class (m, positive down)
    return e.compute(), th[rho].values

def block(da, fac):
    y = dl(da, "yh"); x = dl(da, "xh")
    ny = (da.sizes[y] // fac) * fac; nx = (da.sizes[x] // fac) * fac
    return da.isel({y: slice(0, ny), x: slice(0, nx)}).coarsen({y: fac, x: fac}).mean()

def ape_map(e, R, drho):
    """0.5 * g * drho_i * (e_i - R_i)^2 summed over classes -> J/m2 (2D)."""
    rho = dl(e, "rho2", "rho")
    return (0.5 * G * xr.DataArray(drho, dims=rho) * (e - R) ** 2).sum(dim=rho)

def eke_map(rundir, cut):
    """column-integrated EKE from native snapshots: rho0 * sum_k 0.5*h_k*(var_t u + var_t v) -> J/m2."""
    fs = sorted(f for f in glob.glob(f"{rundir}/output/prog_*.nc")
                if re.match(r".*/prog_\d+\.nc$", f) and int(f.split("_")[-1].split(".")[0]) >= cut)
    maps = []
    for f in fs:
        ds = xr.open_dataset(f, decode_times=False, chunks={"Time": 5})
        u, v, h = ds["u"], ds["v"], ds["h"]
        t = dl(u, "Time")
        uc = 0.5 * (u.isel({dl(u, "xq"): slice(0, -1)}).values + u.isel({dl(u, "xq"): slice(1, None)}).values)
        vc = 0.5 * (v.isel({dl(v, "yq"): slice(0, -1)}).values + v.isel({dl(v, "yq"): slice(1, None)}).values)
        var = 0.5 * (np.nanvar(uc, axis=0) + np.nanvar(vc, axis=0))          # (z,y,x)
        hm = np.nan_to_num(h.mean(dim=t).values)
        maps.append(RHO0 * np.nansum(hm * var, axis=0))
        ds.close()
    m = np.mean(maps, axis=0)
    return m

# --- truth: interfaces, filter to 1/4deg; reference profile from its horizontal mean ---
e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values
w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / \
    (w.sum() * e_tr.sizes[dl(e_tr, "xh")])                                   # R(rho2): common reference
drho = np.gradient(rho2)
e_trf = block(e_tr, FAC)                                                     # filtered-truth interfaces
APE, EKE = {}, {}
APE["truth"] = ape_map(e_trf, R, drho)
# truth EKE filtered: coarsen u,v snapshots then column-int (reuse eke logic quickly, filter first)
fs = sorted(f for f in glob.glob(f"{TRUTH}/output/prog_0*.nc")
            if re.match(r".*/prog_\d+\.nc$", f) and int(f.split("_")[-1].split(".")[0]) >= 10000)
maps = []
for f in fs:
    ds = xr.open_dataset(f, decode_times=False)
    u, v, h = ds["u"], ds["v"], ds["h"]
    t = dl(u, "Time")
    dims = ("Time", "zl", "yh", "xh")
    uc = xr.DataArray(0.5 * (u.isel({dl(u, "xq"): slice(0, -1)}).values +
                             u.isel({dl(u, "xq"): slice(1, None)}).values), dims=dims)
    vc = xr.DataArray(0.5 * (v.isel({dl(v, "yq"): slice(0, -1)}).values +
                             v.isel({dl(v, "yq"): slice(1, None)}).values), dims=dims)
    uc = block(uc, FAC); vc = block(vc, FAC)
    var = 0.5 * (uc.var(dim="Time") + vc.var(dim="Time"))
    hm = block(h.mean(dim=t).fillna(0.0), FAC)
    maps.append((RHO0 * (hm.values * var.values).sum(axis=0)))
    ds.close()
EKE["truth"] = np.mean(maps, axis=0)

for nm, (rd, cut) in RUNS.items():
    e, _ = interfaces(rd, cut)
    APE[nm] = ape_map(e, R, drho)
    EKE[nm] = eke_map(rd, cut)
    print(f"[{nm}] domain-mean APE {float(APE[nm].mean()):.3e}  EKE {np.nanmean(EKE[nm]):.3e} J/m2")
print(f"[truth->1/4] domain-mean APE {float(APE['truth'].mean()):.3e}  EKE {np.nanmean(EKE['truth']):.3e} J/m2")

# --- figure: rows = (truth-np, ANN-np, MEKE-np); cols = dAPE, dEKE ---
rows = [("$\\overline{1/16^\\circ}$ $-$ no-param", "truth"), ("ANN $-$ no-param", "ANN"), ("MEKE $-$ no-param", "MEKE")]
fig, ax = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
xh = APE["no-param"][dl(APE["no-param"], "xh")].values
yh = APE["no-param"][dl(APE["no-param"], "yh")].values
for i, (ttl, key) in enumerate(rows):
    dape = (APE[key].values if hasattr(APE[key], "values") else APE[key]) - APE["no-param"].values
    deke = EKE[key] - EKE["no-param"]
    for j, (fld, vmax, cmap, lbl) in enumerate([(dape, 3e6, plt.cm.PuOr_r, "$\\Delta$APE"),
                                                (deke, 3e4, "RdBu_r", "$\\Delta$EKE")]):
        pc = ax[i, j].pcolormesh(xh, yh, fld, vmin=-vmax, vmax=vmax, cmap=cmap, shading="auto")
        ax[i, j].set_title(f"{lbl}: {ttl}", fontsize=10)
        plt.colorbar(pc, ax=ax[i, j], label="J m$^{-2}$")
for a in ax[-1]: a.set_xlabel("lon")
for a in ax[:, 0]: a.set_ylabel("lat")
fig.suptitle("Channel 1/4°: change in depth-integrated mean APE and resolved EKE relative to no-param\n"
             "(top row = target: what resolving eddies does; APE = rho2-interface displacement vs common truth reference)", fontsize=10)
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"wrote {png}")

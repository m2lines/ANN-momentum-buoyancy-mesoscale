"""Publication versions of the two Sec-4.1 channel figures, written as PDFs directly into the
paper repo (figures/channel_ladder.pdf, figures/channel_ape_eke.pdf) plus PNG previews here.

channel_ladder: restyled combined_ladder.py. FIXES the ACC unit bug -- the `uh` diagnostic is
already m3/s (cell_methods sum), so dividing by RHO0 as combined_ladder.py did understates the
transport by 1035x; true ACC is ~113 Sv.
channel_ape_eke: restyled ape_eke_maps.py (same estimator: rho2-interface-displacement APE vs the
common truth-mean reference; column-integrated snapshot-variance EKE). MJ/kJ units, shared
per-column colorbars, panel letters, no working-figure suptitles. Also prints the regression
slope + pattern r quoted in the paper prose as a cross-check."""
import xarray as xr, numpy as np, glob, re
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])

OUT = "/home/db194/mesoscale_b_ml_parameterization/figures"
PREV = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts"
FAC = 4
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8})

# ============================ Figure 1: ape/eke maps ============================
e_tr, rho2 = interfaces(TRUTH, 10000)
lat = e_tr[dl(e_tr, "yh")].values
w = np.cos(np.deg2rad(lat))
R = (e_tr * xr.DataArray(w, dims=dl(e_tr, "yh"))).sum(dim=[dl(e_tr, "yh"), dl(e_tr, "xh")]) / \
    (w.sum() * e_tr.sizes[dl(e_tr, "xh")])
drho = np.gradient(rho2)
APE, EKE = {}, {}
APE["truth"] = ape_map(block(e_tr, FAC), R, drho)

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
    maps.append(RHO0 * (hm.values * var.values).sum(axis=0))
    ds.close()
EKE["truth"] = np.mean(maps, axis=0)

for nm, (rd, cut) in RUNS.items():
    e, _ = interfaces(rd, cut)
    APE[nm] = ape_map(e, R, drho)
    EKE[nm] = eke_map(rd, cut)

# cross-check the prose numbers: regression slope + pattern r of dAPE on the target
tgt = (APE["truth"].values - APE["no-param"].values).ravel()
for nm in ["ANN", "MEKE"]:
    y = (APE[nm].values - APE["no-param"].values).ravel()
    k = np.isfinite(tgt) & np.isfinite(y)
    print(f"dAPE {nm}: slope {np.polyfit(tgt[k], y[k], 1)[0]:.3f}  r {np.corrcoef(tgt[k], y[k])[0,1]:.3f}")
    ye = (EKE[nm] - EKE["no-param"]).ravel(); te = (EKE["truth"] - EKE["no-param"]).ravel()
    k = np.isfinite(te) & np.isfinite(ye)
    print(f"dEKE {nm}: pattern r {np.corrcoef(te[k], ye[k])[0,1]:.3f}")

xh = APE["no-param"][dl(APE["no-param"], "xh")].values
yh = APE["no-param"][dl(APE["no-param"], "yh")].values
rows = [("Resolved eddies\n(filtered truth)", "truth"), ("ANN", "ANN"), ("MEKE", "MEKE")]
fig, ax = plt.subplots(3, 2, figsize=(7.2, 7.6), sharex=True, sharey=True,
                       constrained_layout=True)
pcs = [None, None]
for i, (rlab, key) in enumerate(rows):
    dape = ((APE[key].values if hasattr(APE[key], "values") else APE[key]) - APE["no-param"].values) / 1e6
    deke = (EKE[key] - EKE["no-param"]) / 1e3
    for j, (fld, vmax, cmap) in enumerate([(dape, 3.0, "PuOr_r"), (deke, 30.0, "RdBu_r")]):
        pcs[j] = ax[i, j].pcolormesh(xh, yh, fld, vmin=-vmax, vmax=vmax, cmap=cmap,
                                     shading="auto", rasterized=True)
        ax[i, j].text(0.015, 0.97, f"({chr(97 + 2*i + j)})", transform=ax[i, j].transAxes,
                      va="top", fontsize=10, fontweight="bold")
    ax[i, 0].set_ylabel("latitude")
    ax[i, 0].annotate(rlab, xy=(0, 0.5), xytext=(-52, 0), xycoords="axes fraction",
                      textcoords="offset points", rotation=90, va="center", ha="center",
                      fontsize=9.5, fontweight="bold")
ax[0, 0].set_title("Change in APE")
ax[0, 1].set_title("Change in resolved EKE")
for a in ax[-1]: a.set_xlabel("longitude")
fig.colorbar(pcs[0], ax=ax[:, 0], orientation="horizontal", shrink=0.85, pad=0.02,
             label="$\\Delta$APE [MJ m$^{-2}$]")
fig.colorbar(pcs[1], ax=ax[:, 1], orientation="horizontal", shrink=0.85, pad=0.02,
             label="$\\Delta$EKE [kJ m$^{-2}$]")
fig.savefig(f"{OUT}/channel_ape_eke.pdf", dpi=300)
fig.savefig(f"{PREV}/channel_ape_eke_preview.png", dpi=150)
print(f"wrote {OUT}/channel_ape_eke.pdf")
plt.close(fig)

# ============================ Figure 2: resolution ladder ============================
def cd(res): return f"{B}/channel_extra_sponge_slow_woc_{res}"
RES = [("1p0", "1$^\\circ$"), ("p5", "1/2$^\\circ$"), ("p25", "1/4$^\\circ$")]
CLOSURES = {
    "no parameterization": ("tau_0.2_cb_0.0_cu_0.0", 4000),
    "ANN ($C_\\mathrm{ANN}=1$)": ("tau_0.2_cb_1.0_cu_0.0_neutral", 11500),
    "MEKE (khf=0.4)": ("tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.4", 11500),
    "MEKE (khf=0.8)": ("tau_0.2_cb_0.0_cu_0.0_MEKE_khf0.8", 11500),
}
TRUTH_LADDER = (f"{cd('p0625')}/tau_0.2_cb_0.0_cu_0.0", 10000)

def twins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]
def acc(f):
    # uh is already m3/s (cell_methods zl:sum yh:sum); combined_ladder.py's extra /RHO0 was a bug
    uh = xr.open_dataset(f, decode_times=False)["uh"]
    return float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / 1e6).mean())
def psi(f):
    vmo = xr.open_dataset(f, decode_times=False)["vmo"]        # kg/s -> /RHO0/1e6 -> Sv
    vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
    P = (vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=dl(vmo, "rho2", "rho", "zl"))
    return float(P.max()), float(P.min())
def eke_scalar(rd, cut):
    v = []
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc")
                    if re.match(r".*/prog_\d+\.nc$", g) and int(g.split("_")[-1].split(".")[0]) >= cut):
        ds = xr.open_dataset(f, decode_times=False, chunks={"zl": 4})
        u, w_ = ds["u"], ds["v"]; t = dl(u, "Time")
        if ds.sizes[t] < 2: ds.close(); continue
        v.append(0.5 * (float(u.var(dim=t).mean().compute()) + float(w_.var(dim=t).mean().compute())))
        ds.close()
    return np.mean(v) if v else np.nan
def metrics(rd, cut):
    wins = [w_ for w_ in twins(rd, cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w_}.nc")]
    if not wins: return None
    a = [acc(f"{rd}/output/prog_tmean_{w_}.nc") for w_ in wins]
    ps = [psi(f"{rd}/output/prog_rho_tmean_{w_}.nc") for w_ in wins]
    return dict(acc=np.mean(a), pmax=np.mean([p[0] for p in ps]),
                pmin=np.mean([p[1] for p in ps]), eke=eke_scalar(rd, cut))

tr = metrics(*TRUTH_LADDER)
print(f"truth: Psi {tr['pmax']:.2f}/{tr['pmin']:.2f} Sv  ACC {tr['acc']:.1f} Sv  EKE {tr['eke']:.3e}")
data = {c: {} for c in CLOSURES}
for c, (sub, cut) in CLOSURES.items():
    for k, (tag, _) in enumerate(RES):
        m = metrics(f"{cd(tag)}/{sub}", cut)
        if m: data[c][k] = m
        print(f"  {c:<22} {tag:>4}: " + ("--" if not m else
              f"Psi {m['pmax']:.2f}/{m['pmin']:.2f}  ACC {m['acc']:.1f}  EKE {m['eke']:.3e}"))

sty = {"no parameterization": ("#8c8c8c", "o"), "ANN ($C_\\mathrm{ANN}=1$)": ("#e6550d", "s"),
       "MEKE (khf=0.4)": ("#4292c6", "^"), "MEKE (khf=0.8)": ("#08519c", "v")}
xs = range(len(RES))
fig, axg = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
ax = axg.ravel()
panels = [("Interior overturning", "pmax", "$\\Psi_\\mathrm{int}$ [Sv]", 1, tr["pmax"]),
          ("Resolved eddy kinetic energy", "eke", "EKE [m$^2$ s$^{-2}$]", 1, tr["eke"]),
          ("ACC transport", "acc", "transport [Sv]", 1, tr["acc"]),
          ("Outcropping-band overturning", "pmin", "$|\\Psi_\\mathrm{out}|$ [Sv]", -1, -tr["pmin"])]
for p, (ttl, key, ylab, sgn, trv) in enumerate(panels):
    a = ax[p]
    a.axhline(trv, color="k", ls="--", lw=1.1, zorder=1,
              label="1/16$^\\circ$ truth" if p == 0 else None)
    for c, (col, mk) in sty.items():
        ys = [sgn * data[c][k][key] if k in data[c] else np.nan for k in xs]
        a.plot(xs, ys, color=col, marker=mk, ms=5, lw=1.4, label=c if p == 0 else None)
    if key == "eke": a.set_yscale("log")
    a.set_xticks(list(xs)); a.set_xticklabels([lbl for _, lbl in RES])
    a.set_title(f"({chr(97+p)}) {ttl}", loc="left")
    a.set_ylabel(ylab)
    a.spines[["top", "right"]].set_visible(False)
for a in axg[-1]: a.set_xlabel("grid spacing")
ax[0].legend(frameon=False, loc="best")
fig.savefig(f"{OUT}/channel_ladder.pdf", dpi=300)
fig.savefig(f"{PREV}/channel_ladder_preview.png", dpi=150)
print(f"wrote {OUT}/channel_ladder.pdf")

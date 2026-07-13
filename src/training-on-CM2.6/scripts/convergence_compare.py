"""1/16 vs 1/32 convergence comparison for the LINEAR-woc channel truth (Part 2 sec 4).
Headline metrics: ACC transport, domain KE, density-space overturning Psi(y,rho). PE->KE conversion
is computed separately+properly (area-weighted column integral of MOM6's PE_to_KE diagnostic, with a
robustness check), since the naive unweighted mean is a noise-dominated cancelling residual.
ANN off => vmo is the total transport (no GM). ponytail: first-pass scalars + one Psi figure."""
import xarray as xr, numpy as np, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO0 = 1035.0
BASE = "/scratch/db194/mom6/feb2026"
RUNS = {
    "1/16 (p0625)":  f"{BASE}/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0",
    "1/32 (p03125)": f"{BASE}/channel_extra_sponge_slow_woc_p03125/tau_0.2_cb_0.0_cu_0.0",
}

def open_stream(outdir, pat):
    fs = sorted(glob.glob(f"{outdir}/{pat}.nc"))
    return (xr.open_mfdataset(fs, combine="by_coords", chunks={"Time": 1}, decode_times=False), fs) if fs else (None, [])

def dim_like(da, *keys):
    for d in da.dims:
        if any(k == d or d.startswith(k) for k in keys): return d
    return None

def mean_En(rundir):
    days, en = [], []
    for line in open(f"{rundir}/ocean.stats"):
        if ", En " not in line: continue
        p = line.split(",")
        try: d = float(p[1]); e = float(p[3].split()[1])
        except (IndexError, ValueError): continue
        days.append(d); en.append(e)
    days, en = np.array(days), np.array(en)
    m = days >= days.min() + 0.4 * (days.max() - days.min())     # drop spin-up
    return float(en[m].mean()), days.min(), days.max()

def pe2ke_proper(files, lat):
    """area-weighted mean of column-integrated PE_to_KE [m3/s3]. Spherical regular grid =>
    cell area ~ cos(lat) (constant factor cancels in the weighted mean)."""
    ds = xr.open_mfdataset(files, combine="by_coords", chunks={"Time": 1}, decode_times=False)
    pk = ds["PE_to_KE"]
    col = pk.mean(dim=dim_like(pk, "Time", "time")).sum(dim=dim_like(pk, "zl", "zL")).values  # (y,x)
    w = np.cos(np.deg2rad(lat))[:, None]                            # area weight ~ cos(lat)
    return float((col * w).sum() / (w * np.ones_like(col)).sum())

out = {}
for name, rundir in RUNS.items():
    o = f"{rundir}/output"; r = {}
    pt, ptf = open_stream(o, "prog_tmean_*")
    uh = pt["uh"]
    sec = uh.sum(dim=[dim_like(uh, "zl"), dim_like(uh, "yh")]) / RHO0 / 1e6
    r["ACC_Sv"] = float(sec.mean().compute())
    r["En"], r["d0"], r["d1"] = mean_En(rundir)

    pr, _ = open_stream(o, "prog_rho_tmean_*")
    vmo = pr["vmo"].mean(dim=dim_like(pr["vmo"], "Time", "time"))
    rho = dim_like(vmo, "rho", "rho2", "zl");
    V = vmo.sum(dim=dim_like(vmo, "xh")) / RHO0 / 1e6
    Psi = V.cumsum(dim=rho).compute()
    r["Psi_pos_Sv"] = float(Psi.max()); r["Psi_neg_Sv"] = float(Psi.min())
    r["_Psi"] = Psi; r["_rho"] = pr[rho].values; r["_yq"] = vmo[dim_like(vmo, "yq")].values

    # proper PE->KE (area-weighted column integral) + per-file robustness
    lat = pt[dim_like(pt["PE_to_KE"], "yh")].values
    r["PE2KE"] = pe2ke_proper(ptf, lat)
    r["PE2KE_byfile"] = [pe2ke_proper([f], lat) for f in ptf]      # robustness across windows
    out[name] = r
    print(f"[done] {name}: ACC={r['ACC_Sv']:.3f}  En={r['En']:.4e}  Psi+={r['Psi_pos_Sv']:.2f} "
          f"Psi-={r['Psi_neg_Sv']:.2f}  PE2KE={r['PE2KE']:.3e}  (days {r['d0']:.0f}-{r['d1']:.0f})")

a, b = list(out.values()); na, nb = list(out.keys())
print("\n" + "="*64); print(f"{'metric':<28}{na:>16}{nb:>16}"); print("-"*64)
def row(lbl, k, f="{:.3f}"): print(f"{lbl:<28}{f.format(a[k]):>16}{f.format(b[k]):>16}")
row("ACC transport [Sv]", "ACC_Sv")
row("domain KE En [m2/s2]", "En", "{:.4e}")
row("Psi interior cell [Sv]", "Psi_pos_Sv")
row("Psi outcrop band [Sv]", "Psi_neg_Sv")
def d(k): return 100*(b[k]-a[k])/abs(a[k])
print("-"*64)
print(f"  Δ(1/32 vs 1/16):  ACC {d('ACC_Sv'):+.1f}%   En {d('En'):+.1f}%   "
      f"Psi_interior {d('Psi_pos_Sv'):+.1f}%   Psi_outcrop {d('Psi_neg_Sv'):+.1f}%")
print("="*64)
print("\nPE->KE conversion (area-weighted column integral, m3/s3) — separate, robustness-checked:")
for nm, r in out.items():
    bw = "  ".join(f"{x:.2e}" for x in r["PE2KE_byfile"])
    print(f"  {nm:<16} combined={r['PE2KE']:.3e}   per-window=[{bw}]")
print(f"  Δ(1/32 vs 1/16) = {d('PE2KE'):+.1f}%")

# Psi(y,rho) figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (nm, r) in zip(axes, out.items()):
    P = r["_Psi"]
    pc = ax.pcolormesh(r["_yq"], r["_rho"], P.values, cmap="RdBu_r",
                       vmin=-abs(P).max(), vmax=abs(P).max(), shading="auto")
    ax.set_title(f"{nm}   interior +{r['Psi_pos_Sv']:.1f} / outcrop {r['Psi_neg_Sv']:.1f} Sv")
    ax.set_xlabel("latitude"); ax.invert_yaxis(); plt.colorbar(pc, ax=ax, label="Psi [Sv]")
axes[0].set_ylabel("rho2 [kg/m3]")
fig.suptitle("Residual overturning Psi(y,rho) — resolution convergence (LINEAR woc, tau=0.2, no-param)")
fig.tight_layout()
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/convergence_Psi.png"
fig.savefig(png, dpi=130, bbox_inches="tight"); print(f"\nwrote {png}")

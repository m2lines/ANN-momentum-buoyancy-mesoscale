"""GELU vs ReLU online comparison (same binary, activation is the only difference; Part 2 sec 4 test).
Metrics on settled windows: ACC transport, interior/outcrop Psi, resolved EKE; plus the 1/4deg APE-removal
slope vs the filtered-truth target. Includes a backward-compat anchor: the new-binary ReLU run vs the
existing neutral run (old Feb-9 binary) should agree."""
import numpy as np, xarray as xr, glob, re
RHO0, G = 1035.0, 9.8
B = "/scratch/db194/mom6/feb2026"
def dl(da, *k):
    for d in da.dims:
        if any(x == d or d.startswith(x) for x in k): return d
def wins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]
def acc(f):
    uh = xr.open_dataset(f, decode_times=False)["uh"]; return float((uh.sum(dim=[dl(uh,"zl"),dl(uh,"yh")])/RHO0/1e6).mean())
def psi(f):
    v = xr.open_dataset(f, decode_times=False)["vmo"]; v = v.mean(dim=dl(v,"Time")) if dl(v,"Time") else v
    P = (v.sum(dim=dl(v,"xh"))/RHO0/1e6).cumsum(dim=dl(v,"rho2","rho","zl")); return float(P.max()), float(P.min())
def eke(rd, cut):
    vals=[]
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$",g) and int(g.split("_")[-1].split(".")[0])>=cut):
        ds=xr.open_dataset(f,decode_times=False); u,w=ds["u"],ds["v"]; t=dl(u,"Time"); vals.append(0.5*(float(u.var(dim=t).mean())+float(w.var(dim=t).mean())))
    return np.mean(vals) if vals else np.nan
def M(rd, cut):
    ws=[w for w in wins(rd,cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    a=[acc(f"{rd}/output/prog_tmean_{w}.nc") for w in ws]; ps=[psi(f"{rd}/output/prog_rho_tmean_{w}.nc") for w in ws]
    return dict(acc=np.mean(a),acc_s=np.std(a),pmax=np.mean([p[0] for p in ps]),pmin=np.mean([p[1] for p in ps]),eke=eke(rd,cut),n=len(ws))

RUNS = {
 "1deg ReLU": (f"{B}/channel_extra_sponge_slow_woc_1p0/tau_0.2_cb_1.0_cu_0.0_act_relu", 11500),
 "1deg GELU": (f"{B}/channel_extra_sponge_slow_woc_1p0/tau_0.2_cb_1.0_cu_0.0_act_gelu", 11500),
 "1/4 ReLU":  (f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_act_relu", 11500),
 "1/4 GELU":  (f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_act_gelu", 11500),
 "1/4 ReLU(old bin, existing)": (f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_neutral", 11500),
}
r = {k: M(rd, c) for k, (rd, c) in RUNS.items()}
truth = M(f"{B}/channel_extra_sponge_slow_woc_p0625/tau_0.2_cb_0.0_cu_0.0", 10000)
print(f"{'run':<30}{'ACC':>9}{'Psi_int':>9}{'Psi_out':>9}{'EKE':>11}{'n':>3}")
for k, m in r.items():
    print(f"{k:<30}{m['acc']:.4f}{'':>2}{m['pmax']:>8.2f}{m['pmin']:>9.2f}{m['eke']:>11.3e}{m['n']:>3}")
print(f"{'truth 1/16':<30}{truth['acc']:.4f}{'':>2}{truth['pmax']:>8.2f}{truth['pmin']:>9.2f}{truth['eke']:>11.3e}")

def delta(a, b, key):
    va, vb = r[a][key], r[b][key]; return 100*(vb-va)/abs(va)
print("\n=== GELU - ReLU (same binary; activation only) ===")
for res, a, g in [("1deg", "1deg ReLU", "1deg GELU"), ("1/4deg", "1/4 ReLU", "1/4 GELU")]:
    print(f"{res}: ACC {delta(a,g,'acc'):+.1f}%  Psi_int {delta(a,g,'pmax'):+.1f}%  Psi_out {delta(a,g,'pmin'):+.1f}%  EKE {delta(a,g,'eke'):+.1f}%")
print("\n=== backward-compat: new-binary ReLU vs existing neutral run (old binary) at 1/4deg ===")
for key in ("acc", "pmax", "pmin", "eke"):
    print(f"  {key}: new {r['1/4 ReLU'][key]:.4g}  vs existing {r['1/4 ReLU(old bin, existing)'][key]:.4g}  ({delta('1/4 ReLU(old bin, existing)','1/4 ReLU',key):+.1f}%)")

# --- 1/4deg APE-removal slope vs filtered-truth target, ReLU vs GELU ---
print("\n=== 1/4deg APE-removal slope (vs filtered-truth target) ===")
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])
e_tr, rho2 = interfaces(TRUTH, 10000); lat = e_tr[dl(e_tr,"yh")].values; w = np.cos(np.deg2rad(lat))
R = (e_tr*xr.DataArray(w,dims=dl(e_tr,"yh"))).sum(dim=[dl(e_tr,"yh"),dl(e_tr,"xh")])/(w.sum()*e_tr.sizes[dl(e_tr,"xh")]); drho=np.gradient(rho2)
A_tr = ape_map(block(e_tr,4),R,drho); e_np,_=interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0",4000); A_np=ape_map(e_np,R,drho)
tgt=(A_tr.values-A_np.values).ravel()
for lbl, sub in [("ReLU","act_relu"),("GELU","act_gelu")]:
    e,_=interfaces(f"{P25}/tau_0.2_cb_1.0_cu_0.0_{sub}",11500); dA=(ape_map(e,R,drho).values-A_np.values).ravel()
    m=np.isfinite(dA)&np.isfinite(tgt); s=np.polyfit(tgt[m],dA[m],1)[0]; rr=np.corrcoef(dA[m],tgt[m])[0,1]
    print(f"  {lbl}: slope {s:.2f}  pattern r {rr:.2f}")

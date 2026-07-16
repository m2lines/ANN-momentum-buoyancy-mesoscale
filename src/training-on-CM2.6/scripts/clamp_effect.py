"""Isolate the Upsilon/flux-clamp effect (Part 2 sec 4): new binary, neutral ReLU model, warm-start --
the ONLY difference is whether MESO_UPSILON_CLAMP(15)/FLUX_CLAMP/MAG_GRAD_FLOOR are active (clamp-on)
or disabled (clamp-off). Old-binary neutral run = independent unclamped anchor to validate the diagnosis."""
import numpy as np, xarray as xr, glob, re
RHO0, G = 1035.0, 9.8
B = "/scratch/db194/mom6/feb2026"
def dl(da,*k):
    for d in da.dims:
        if any(x==d or d.startswith(x) for x in k): return d
def wins(rd,cut):
    ws=sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc")); return [f"{w:06d}" for w in ws if w>=cut]
def acc(f):
    uh=xr.open_dataset(f,decode_times=False)["uh"]; return float((uh.sum(dim=[dl(uh,"zl"),dl(uh,"yh")])/RHO0/1e6).mean())
def psi(f):
    v=xr.open_dataset(f,decode_times=False)["vmo"]; v=v.mean(dim=dl(v,"Time")) if dl(v,"Time") else v
    P=(v.sum(dim=dl(v,"xh"))/RHO0/1e6).cumsum(dim=dl(v,"rho2","rho","zl")); return float(P.max()),float(P.min())
def eke(rd,cut):
    vv=[]
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$",g) and int(g.split("_")[-1].split(".")[0])>=cut):
        ds=xr.open_dataset(f,decode_times=False); u,w=ds["u"],ds["v"]; t=dl(u,"Time"); vv.append(0.5*(float(u.var(dim=t).mean())+float(w.var(dim=t).mean())))
    return np.mean(vv) if vv else np.nan
def Mt(rd,cut):
    ws=[w for w in wins(rd,cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    a=[acc(f"{rd}/output/prog_tmean_{w}.nc") for w in ws]; ps=[psi(f"{rd}/output/prog_rho_tmean_{w}.nc") for w in ws]
    return dict(acc=np.mean(a),pmax=np.mean([p[0] for p in ps]),pmin=np.mean([p[1] for p in ps]),eke=eke(rd,cut),n=len(ws))

RUNS = {
 "1deg clamp-ON":  f"{B}/channel_extra_sponge_slow_woc_1p0/tau_0.2_cb_1.0_cu_0.0_act_relu",
 "1deg clamp-OFF": f"{B}/channel_extra_sponge_slow_woc_1p0/tau_0.2_cb_1.0_cu_0.0_clampoff",
 "1/4 clamp-ON":   f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_act_relu",
 "1/4 clamp-OFF":  f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_clampoff",
 "1/4 OLD-binary (unclamped anchor)": f"{B}/channel_extra_sponge_slow_woc_p25/tau_0.2_cb_1.0_cu_0.0_neutral",
}
r={k:Mt(rd,11500) for k,rd in RUNS.items()}
print(f"{'run':<38}{'ACC':>9}{'Psi_int':>9}{'Psi_out':>9}{'EKE':>11}{'n':>3}")
for k,m in r.items(): print(f"{k:<38}{m['acc']:.4f}{'':>2}{m['pmax']:>8.2f}{m['pmin']:>9.2f}{m['eke']:>11.3e}{m['n']:>3}")
def d(a,b,key): return 100*(r[b][key]-r[a][key])/abs(r[a][key])
print("\n=== clamp effect = clamp-ON minus clamp-OFF (turning the Upsilon clamp ON does this) ===")
for res,on,off in [("1deg","1deg clamp-ON","1deg clamp-OFF"),("1/4deg","1/4 clamp-ON","1/4 clamp-OFF")]:
    print(f"{res}: ACC {d(off,on,'acc'):+.1f}%  Psi_int {d(off,on,'pmax'):+.1f}%  Psi_out {d(off,on,'pmin'):+.1f}%  EKE {d(off,on,'eke'):+.1f}%")

# --- APE-removal slope: clamp-on vs clamp-off vs old-binary (does clamp-off recover 0.91?) ---
print("\n=== 1/4deg APE-removal slope (validates the clamp is THE drift) ===")
exec(open("/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/ape_eke_maps.py").read().split("# --- truth")[0])
e_tr,rho2=interfaces(TRUTH,10000); lat=e_tr[dl(e_tr,"yh")].values; w=np.cos(np.deg2rad(lat))
R=(e_tr*xr.DataArray(w,dims=dl(e_tr,"yh"))).sum(dim=[dl(e_tr,"yh"),dl(e_tr,"xh")])/(w.sum()*e_tr.sizes[dl(e_tr,"xh")]); drho=np.gradient(rho2)
A_tr=ape_map(block(e_tr,4),R,drho); e_np,_=interfaces(f"{P25}/tau_0.2_cb_0.0_cu_0.0",4000); A_np=ape_map(e_np,R,drho); tgt=(A_tr.values-A_np.values).ravel()
for lbl,rd in [("clamp-ON",f"{P25}/tau_0.2_cb_1.0_cu_0.0_act_relu"),("clamp-OFF",f"{P25}/tau_0.2_cb_1.0_cu_0.0_clampoff"),("OLD-binary",f"{P25}/tau_0.2_cb_1.0_cu_0.0_neutral")]:
    e,_=interfaces(rd,11500); dA=(ape_map(e,R,drho).values-A_np.values).ravel(); m=np.isfinite(dA)&np.isfinite(tgt)
    print(f"  {lbl:<12} slope {np.polyfit(tgt[m],dA[m],1)[0]:.2f}  pattern r {np.corrcoef(dA[m],tgt[m])[0,1]:.2f}")

"""Const-GM retuning test (Part 2 sec 4): does a fixed kappa tuned at 1deg hold at finer resolution,
or must GM be retuned? kappa in {500,1000,2000} x {1deg, 1/2deg, 1/4deg}, warm-started like the rest of
the ladder. Also: does const-GM kill resolved EKE like MEKE does? Truth = 1/16."""
import xarray as xr, numpy as np, glob, re

RHO0 = 1035.0
B = "/scratch/db194/mom6/feb2026"
def cd(res): return f"{B}/channel_extra_sponge_slow_woc_{res}"
RES = [("1p0", "1deg"), ("p5", "1/2deg"), ("p25", "1/4deg")]
KAPPA = [500, 1000, 2000]
CUT = 11500

def dl(da, *k):
    for d in da.dims:
        if any(x == d or d.startswith(x) for x in k): return d
    return None
def twins(rd, cut):
    ws = sorted(int(f.split("_")[-1].split(".")[0]) for f in glob.glob(f"{rd}/output/prog_tmean_*.nc"))
    return [f"{w:06d}" for w in ws if w >= cut]
def acc(f):
    uh = xr.open_dataset(f, decode_times=False)["uh"]; return float((uh.sum(dim=[dl(uh, "zl"), dl(uh, "yh")]) / RHO0 / 1e6).mean())
def psimax(f):
    vmo = xr.open_dataset(f, decode_times=False)["vmo"]; vmo = vmo.mean(dim=dl(vmo, "Time")) if dl(vmo, "Time") else vmo
    return float((vmo.sum(dim=dl(vmo, "xh")) / RHO0 / 1e6).cumsum(dim=dl(vmo, "rho2", "rho", "zl")).max())
def eke(rd, cut):
    v = []
    for f in sorted(g for g in glob.glob(f"{rd}/output/prog_*.nc") if re.match(r".*/prog_\d+\.nc$", g) and int(g.split("_")[-1].split(".")[0]) >= cut):
        ds = xr.open_dataset(f, decode_times=False); u, w = ds["u"], ds["v"]; t = dl(u, "Time")
        v.append(0.5 * (float(u.var(dim=t).mean()) + float(w.var(dim=t).mean())))
    return np.mean(v) if v else np.nan
def metrics(rd, cut):
    wins = [w for w in twins(rd, cut) if glob.glob(f"{rd}/output/prog_rho_tmean_{w}.nc")]
    if not wins: return None
    return dict(acc=np.mean([acc(f"{rd}/output/prog_tmean_{w}.nc") for w in wins]),
                psi=np.mean([psimax(f"{rd}/output/prog_rho_tmean_{w}.nc") for w in wins]),
                eke=eke(rd, cut), n=len(wins))

tr = metrics(f"{cd('p0625')}/tau_0.2_cb_0.0_cu_0.0", 10000)
print(f"truth 1/16: Psi {tr['psi']:.2f}  ACC {tr['acc']:.3f}  EKE {tr['eke']:.3e}\n")
print(f"{'res':<8}{'kappa':>7}{'Psi_int':>9}{'ACC':>8}{'EKE':>11}{'EKE/noparam':>12}")
NOPAR = {}
for tag, lab in RES:
    NOPAR[tag] = metrics(f"{cd(tag)}/tau_0.2_cb_0.0_cu_0.0", 4000)
for tag, lab in RES:
    for K in KAPPA:
        m = metrics(f"{cd(tag)}/tau_0.2_cb_0.0_cu_0.0_GM_kh{K}", CUT)
        s = "--" if not m else f"{m['psi']:>9.2f}{m['acc']:>8.3f}{m['eke']:>11.3e}{m['eke']/NOPAR[tag]['eke']*100:>11.0f}%"
        print(f"{lab:<8}{K:>7}{s}")
    print(f"{lab:<8}{'(np)':>7}{NOPAR[tag]['psi']:>9.2f}{NOPAR[tag]['acc']:>8.3f}{NOPAR[tag]['eke']:>11.3e}{'100%':>12}")

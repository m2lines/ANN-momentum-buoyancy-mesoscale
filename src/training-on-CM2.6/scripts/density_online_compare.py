"""Online density-reference comparison for the 1deg channel (Part 2 sec 4 de-risk).
All runs: cb=1, tau=0.2, warm-started from the SAME day-10000 sigma0 restart, only the ANN differs.
  Pair A (deployed nets): EXP0 (sigma0, all4)   vs  EXP_neutral_all4 (neutral, all4)
  Pair B (matched):       EXP_sigma0 [4,9,12]    vs  EXP_neutral [4,9,12]   (density-ONLY difference)
Pair A conflates density with the network (two independent trainings); pair B isolates density.
Metrics per clean 2000-day window: ACC transport, residual overturning Psi(y,rho); domain KE from ocean.stats.
Runs whose windows are not on disk yet are skipped, so this can be re-run as jobs land."""
import xarray as xr, numpy as np, glob

RHO0 = 1035.0
BASE = "/scratch/db194/mom6/feb2026/channel_extra_sponge_slow_woc_1p0"
# label -> (rundir, clean settled window-index strings)
RUNS = {
    "A sigma0-all4 (EXP0)":  (f"{BASE}/tau_0.2_cb_1.0_cu_0.0",               ["004050", "006050", "008050"]),
    "A neutral-all4":        (f"{BASE}/tau_0.2_cb_1.0_cu_0.0_neutral_cont",  ["012050", "014050", "016050"]),
    "B matched-sigma0":      (f"{BASE}/tau_0.2_cb_1.0_cu_0.0_matched_sigma0", ["012050", "014050", "016050"]),
    "B matched-neutral":     (f"{BASE}/tau_0.2_cb_1.0_cu_0.0_matched_neutral", ["012050", "014050", "016050"]),
}

def dim_like(da, *keys):
    for d in da.dims:
        if any(k == d or d.startswith(k) for k in keys): return d
    return None

def acc_window(f):
    ds = xr.open_dataset(f, decode_times=False); uh = ds["uh"]
    sec = uh.sum(dim=[dim_like(uh, "zl"), dim_like(uh, "yh")]) / RHO0 / 1e6   # Sv, per (Time,xq)
    return float(sec.mean())

def psi_window(f):
    ds = xr.open_dataset(f, decode_times=False)
    vmo = ds["vmo"].mean(dim=dim_like(ds["vmo"], "Time", "time"))
    rho = dim_like(vmo, "rho2", "rho", "zl")
    V = vmo.sum(dim=dim_like(vmo, "xh")) / RHO0 / 1e6
    Psi = V.cumsum(dim=rho)
    return float(Psi.max()), float(Psi.min())            # interior cell (+), outcrop band (-)

def En_over(rundir, windows):
    lo = min(int(w) for w in windows); hi = max(int(w) for w in windows) + 2000
    e = []
    for L in open(f"{rundir}/ocean.stats"):
        if ", En " not in L: continue
        p = L.split(",")
        try:
            d = float(p[1])
            if lo <= d < hi: e.append(float(p[3].split()[1]))
        except (IndexError, ValueError): pass
    e = np.array(e); return e.mean(), e.std()

def stat(v): return f"{np.mean(v):.4g} +/- {np.std(v):.2g}"

out = {}
for name, (rundir, wins) in RUNS.items():
    accs, psip, psim = [], [], []
    have = []
    for w in wins:
        pt = glob.glob(f"{rundir}/output/prog_tmean_{w}.nc")
        pr = glob.glob(f"{rundir}/output/prog_rho_tmean_{w}.nc")
        if not pt or not pr: continue
        have.append(w)
        accs.append(acc_window(pt[0]))
        a, b = psi_window(pr[0]); psip.append(a); psim.append(b)
    if not have:
        print(f"[skip] {name}: no windows on disk yet"); continue
    en_m, en_s = En_over(rundir, have)
    out[name] = dict(acc=accs, psip=psip, psim=psim, en=(en_m, en_s), n=len(have))
    print(f"[{name}]  windows={have}")
    print(f"    ACC   {stat(accs)} Sv")
    print(f"    Psi+  {stat(psip)} Sv   Psi- {stat(psim)} Sv")
    print(f"    En    {en_m:.4e} +/- {en_s:.1e} m2/s2")

def pair(nsig, nneu, tag):
    if nsig not in out or nneu not in out: return
    s, n = out[nsig], out[nneu]
    print(f"\n=== {tag}:  {nneu}  vs  {nsig} ===")
    for k, lbl in [("acc", "ACC transport"), ("psip", "Psi interior"), ("psim", "Psi outcrop"), ("en", "domain KE")]:
        sv = s[k][0] if k == "en" else np.mean(s[k])
        nv = n[k][0] if k == "en" else np.mean(n[k])
        d = 100 * (nv - sv) / abs(sv)
        print(f"    {lbl:<16} sigma0 {sv:.4g}   neutral {nv:.4g}   Delta {d:+.1f}%")

print("\n" + "=" * 66)
pair("A sigma0-all4 (EXP0)", "A neutral-all4", "PAIR A (deployed nets: density + network)")
pair("B matched-sigma0", "B matched-neutral", "PAIR B (matched: density ONLY)")
print("=" * 66)

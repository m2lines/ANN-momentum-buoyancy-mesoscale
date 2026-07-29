"""How long must an NW2 run be before the closure's response has emerged?

Dhruv's question: 30 000-day legs are heavy -- can 5-10 years already show the response? The
dec2025 R2 runs all branch from the same spun-up state and saved 30 sequential 1000-day windows,
so the adjustment is directly measurable rather than a matter of judgement.

For each closure and each window we compute cheap proxies of the three axes:
  KE        -- domain-mean kinetic energy (the EKE-preservation axis)
  |e| rms   -- rms interface displacement about the run's own final state (mean-state axis)
  ACC       -- zonally-integrated zonal transport (circulation axis)
and report, for each metric, how many windows are needed before a run sits within 10% and 5% of
its own final value -- i.e. when the response has effectively saturated."""
import numpy as np, xarray as xr, glob, re

B = "/scratch/db194/mom6/dec2025/NeverWorld2/R2-WENO"
RUNS = ["ANN_c1p0", "ANN_c2p0", "GM200", "GM800", "MEKE"]
RHO0 = 1035.0

def windows(run):
    f = sorted(glob.glob(f"{B}/{run}/longmean_*.nc")) or sorted(glob.glob(f"{B}/{run}/output/longmean_*.nc"))
    return [(int(re.search(r"longmean_0*(\d+)\.nc", x).group(1)), x) for x in f]

series = {}
for run in RUNS:
    w = windows(run)
    if not w:
        print(f"{run}: no windows"); continue
    days, ke, erms, acc = [], [], [], []
    for d, f in w:
        ds = xr.open_dataset(f, decode_times=False)
        t = "time" if "time" in ds.dims else None
        g = lambda v: (ds[v].mean(t) if t else ds[v])
        k = float(np.nanmean(g("KE").values))
        if not np.isfinite(k):                    # some final windows are written all-NaN
            ds.close(); continue
        ke.append(k)
        erms.append(np.nan_to_num(g("e").values))        # keep, differenced below
        acc.append(float(np.nansum(g("uh").values, axis=(0, 1)).mean() / RHO0 / 1e6))
        days.append(d)
        ds.close()
    E = np.stack(erms)                                    # (nwin, zi, y, x)
    e_dev = np.sqrt(np.nanmean((E - E[-1]) ** 2, axis=(1, 2, 3)))   # rms vs the run's own final state
    series[run] = dict(days=np.array(days), KE=np.array(ke), edev=e_dev, ACC=np.array(acc))
    print(f"{run}: {len(days)} windows, days {days[0]}-{days[-1]}", flush=True)

print(f"\n{'run':<10}{'metric':<8}{'final':>12}   windows(=1000 d) to reach within 10% / 5% of final")
for run, s in series.items():
    d0 = s["days"][0]
    for m in ("KE", "ACC"):
        v = s[m]; fin = v[-1]
        rel = np.abs(v - fin) / max(abs(fin), 1e-30)
        def first_within(tol):
            ok = np.where(rel <= tol)[0]
            # first index after which it STAYS within tol
            for i in ok:
                if np.all(rel[i:] <= tol): return s["days"][i] - d0 + 1000
            return None
        a, b = first_within(0.10), first_within(0.05)
        print(f"{run:<10}{m:<8}{fin:>12.4g}   {str(a):>8} d / {str(b):>8} d"
              f"   ({'' if a is None else f'{a/365:.0f} yr'} / {'' if b is None else f'{b/365:.0f} yr'})")
    e = s["edev"]; sc = e[0] if e[0] > 0 else 1.0
    idx = np.where(e / sc <= 0.10)[0]
    stay = next((s["days"][i] - d0 + 1000 for i in idx if np.all(e[i:] / sc <= 0.10)), None)
    print(f"{run:<10}{'e_rms':<8}{'-':>12}   interface drift falls to 10% of its initial excursion "
          f"after {stay} d ({'' if stay is None else f'{stay/365:.0f} yr'})")

print("\nDivergence between closures (the quantity a comparison actually needs):")
if "ANN_c1p0" in series and "GM200" in series:
    a, g = series["ANN_c1p0"], series["GM200"]
    n = min(len(a["days"]), len(g["days"]))
    dk = np.abs(a["KE"][:n] - g["KE"][:n]); dk_fin = dk[-1]
    for tol in (0.5, 0.8, 0.9):
        i = np.argmax(dk >= tol * dk_fin)
        print(f"  ANN-vs-GM200 KE separation reaches {tol*100:.0f}% of its final value after "
              f"{a['days'][i] - a['days'][0] + 1000} d ({(a['days'][i]-a['days'][0]+1000)/365:.1f} yr)")

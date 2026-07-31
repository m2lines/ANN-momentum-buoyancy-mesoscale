"""Audit the dec2025 NW2 runs BEFORE any are deleted (Dhruv: "peer into the data first").

823 GB across R2-WENO / R3 / R4, all using the old sigma0 network `ann_instance_20Dec.nc`. Two
questions: (1) do their bulk energetics look sane, i.e. were these runs healthy, and (2) is there
anything in them worth keeping once the neutral-network re-runs exist?

Everything here comes from ocean.stats.nc, which is a few hundred kB per run -- so the audit itself
costs nothing and can be re-read after the bulk data is gone. For each run we record the final APE
and KE, their drift over the record, velocity truncations (numerical distress), and max CFL."""
import numpy as np, xarray as xr, glob, os

ROOTS = ["/scratch/db194/mom6/dec2025/NeverWorld2"]
rows = []
for root in ROOTS:
    for st in sorted(glob.glob(f"{root}/*/*/ocean.stats.nc")) + sorted(glob.glob(f"{root}/*/*/output/ocean.stats.nc")):
        run = st.replace(root + "/", "").replace("/output/ocean.stats.nc", "").replace("/ocean.stats.nc", "")
        try:
            d = xr.open_dataset(st, decode_times=False)
        except Exception as e:
            rows.append((run, None, str(e)[:40])); continue
        t = d["Time"].values
        ape = d["APE"].values.sum(axis=1); ke = d["KE"].values.sum(axis=1)
        ntr = d["Ntrunc"].values if "Ntrunc" in d else np.zeros_like(t)
        cfl = d["max_CFL_trans"].values if "max_CFL_trans" in d else np.zeros_like(t)
        n = max(len(t)//10, 1)
        sz = sum(os.path.getsize(f) for f in glob.glob(os.path.dirname(st)+"/*.nc")) / 1e9
        rows.append((run, dict(days=(t[0], t[-1]), nrec=len(t),
                               ape=ape[-n:].mean(), ape_drift=(ape[-n:].mean()-ape[:n].mean())/ape[:n].mean(),
                               ke=ke[-n:].mean(), ke_drift=(ke[-n:].mean()-ke[:n].mean())/ke[:n].mean(),
                               ntr=int(ntr.sum()), cfl=float(np.nanmax(cfl)), gb=sz), None))
        d.close()

print(f"{'run':<26}{'days':>16}{'nrec':>6}{'APE[1e20]':>11}{'drift%':>8}{'KE[1e17]':>10}{'drift%':>8}"
      f"{'Ntrunc':>8}{'maxCFL':>8}{'GB':>7}")
for run, r, err in rows:
    if r is None:
        print(f"{run:<26}  ERROR {err}"); continue
    print(f"{run:<26}{f'{r[chr(100)+chr(97)+chr(121)+chr(115)][0]:.0f}-{r[chr(100)+chr(97)+chr(121)+chr(115)][1]:.0f}':>16}"
          f"{r['nrec']:>6}{r['ape']/1e20:>11.3f}{100*r['ape_drift']:>8.1f}{r['ke']/1e17:>10.3f}"
          f"{100*r['ke_drift']:>8.1f}{r['ntr']:>8}{r['cfl']:>8.2f}{r['gb']:>7.1f}")

print("\nverdict guide: |drift| < ~5% over the record = equilibrated; Ntrunc >> 100 or CFL > 0.5 =")
print("numerical distress; a healthy NW2 R2 sits near APE 1.0e20 J, KE 3-4e17 J.")

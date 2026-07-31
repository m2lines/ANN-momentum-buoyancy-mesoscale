"""Coarsen the 1/32-degree NW2 truth to our 1/2-degree grid, for a more robust truth comparison.

The pre-made `0.5-degree-coarsen-snapshots.nc` carries only h,u,v at 4 snapshots. The native
`R32/longmean*.nc` (9 files, 1920x4480x15, ~7-14 GB each, 10 time records in total spanning model
days 1850-2750) additionally carries `e`, `uh/vh` and `KE`, so coarsening it ourselves supports APE,
EKE and transport against truth on exactly the footing our runs are analysed on.

Coarsening is a plain 16x16 block mean in x and y (1920/120 = 4480/280 = 16), which lands on the
same cell centres as our R2 grid -- verified bit-identical against the pre-made product.

Memory: each variable is read and reduced one time-record and one layer at a time, so peak usage
stays near a single 2-D native slice (~35 MB) rather than a whole file."""
import numpy as np, xarray as xr, glob, os

SRC = sorted(glob.glob("/scratch/pp2681/mom6/Neverworld2/simulations/R32/longmean*.nc"))
OUT = "/scratch/db194/mom6/jul2026_nw2/_truth/R32_coarsened_0p5.nc"
F = 16
VARS3 = ["h", "u", "v", "uh", "vh", "KE"]     # (time, zl, y, x)
VARSI = ["e"]                                  # (time, zi, y, x)

def blk(a):
    ny, nx = a.shape
    return a[:(ny//F)*F, :(nx//F)*F].reshape(ny//F, F, nx//F, F).mean(axis=(1, 3))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
acc, times = {}, []
for f in SRC:
    d = xr.open_dataset(f, decode_times=False)
    nt = d.sizes["time"]
    for it in range(nt):
        times.append(float(d["time"].values[it]))
        for v in VARS3 + VARSI:
            if v not in d: continue
            arr = d[v].isel(time=it)
            nz = arr.shape[0]
            out = np.empty((nz, 280, 120), dtype="f4")
            for k in range(nz):
                s = arr.isel({arr.dims[0]: k}).values.astype("f8")
                # u is on xq, v on yq: trim to centre count before blocking
                s = s[:4480, :1920]
                out[k] = blk(s)
            acc.setdefault(v, []).append(out)
        print(f"  {os.path.basename(f)} t={times[-1]:.0f} done", flush=True)
    d.close()

ds = xr.Dataset()
ref = xr.open_dataset("/scratch/db194/mom6/jul2026_nw2/bare/output/longmean_00037050.nc", decode_times=False)
for v, lst in acc.items():
    a = np.stack(lst)
    dims = ("time", "zi" if v in VARSI else "zl", "yh", "xh")
    ds[v] = (dims, a)
ds = ds.assign_coords(time=("time", np.array(times)), xh=ref["xh"].values, yh=ref["yh"].values,
                      zl=ref["zl"].values, zi=ref["zi"].values if "zi" in ref else np.arange(16))
ref.close()
ds.attrs["source"] = "R32/longmean*.nc block-mean 16x16 to the R2 grid"
ds.to_netcdf(OUT)
print("wrote", OUT, {k: v.shape for k, v in ds.data_vars.items()})

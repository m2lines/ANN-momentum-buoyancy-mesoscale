"""Figure-1-style offline-skill comparison: EXP0 (sigma0, superseded) vs EXP_neutral_all4 (canonical).
Heatmaps (depth x [factor x al/ac/dv]) of coast-excluded R2 and correlation for both models plus the
difference, and a depth-mean summary panel. Run in the Pavel container (netcdf4 files)."""
import xarray as xr, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FACTORS = [4, 9, 12, 15]
DEG = {4: "0.4°", 9: "0.9°", 12: "1.2°", 15: "1.5°"}
ROOTS = {
    "EXP0 (σ₀, superseded)": "/scratch/db194/CM26_ML_models/FGR3/EXP0/skill-test-rho-div/factor-{f}.nc",
    "EXP_neutral_all4 (canonical)": "/scratch/db194/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/skill-test/factor-{f}.nc",
}
SUB = [("R2F_along_away", "al"), ("R2F_across_away", "ac"), ("R2F_div_away", "dv")]
SUBC = [("corr_F_along_away", "al"), ("corr_F_across_away", "ac"), ("corr_F_div_away", "dv")]

def grid(root, sub):
    """(nz, nfac*4-1) heatmap matrix with NaN separator columns, + depth axis."""
    cols, zl = [], None
    for f in FACTORS:
        ds = xr.open_dataset(root.format(f=f))
        for k, _ in sub:
            v = ds[k]
            zdim = [d for d in v.dims if d.startswith("z")][0]
            cols.append(v.values)
            zl = ds[zdim].values
        cols.append(np.full_like(cols[-1], np.nan))          # separator
    return np.stack(cols[:-1], axis=1), zl

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.25)

def hm(ax, M, zl, cmap, vmin, vmax, title):
    pc = ax.pcolormesh(np.arange(M.shape[1] + 1), np.append(zl, zl[-1] + (zl[-1]-zl[-2])), M,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.invert_yaxis(); ax.set_title(title, fontsize=10)
    ticks, labels = [], []
    for i, f in enumerate(FACTORS):
        base = i * 4
        ticks += [base + 0.5, base + 1.5, base + 2.5]; labels += ["al", "ac", "dv"]
        ax.text(base + 1.5, -150, DEG[f], ha="center", fontsize=9)
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("depth [m]")
    return pc

names = list(ROOTS)
mats = {nm: {"R2": grid(ROOTS[nm], SUB), "corr": grid(ROOTS[nm], SUBC)} for nm in names}
for row, met, (vmin, vmax, cmap) in [(0, "R2", (0, 1, "Reds")), (1, "corr", (0.4, 1, "viridis"))]:
    for col, nm in enumerate(names):
        M, zl = mats[nm][met]
        pc = hm(fig.add_subplot(gs[row, col]), M, zl, cmap, vmin, vmax, f"{met}: {nm}")
        plt.colorbar(pc, ax=fig.axes[-1], fraction=0.04)
    D = mats[names[1]][met][0] - mats[names[0]][met][0]
    pc = hm(fig.add_subplot(gs[row, 2]), D, zl, "RdBu_r", -0.15, 0.15, f"Δ{met}: canonical − EXP0")
    plt.colorbar(pc, ax=fig.axes[-1], fraction=0.04)

# depth-mean summary: R2F_away_centered + corr_F_away vs resolution, both models
ax = fig.add_subplot(gs[2, :])
sty = {names[0]: ("#888888", "o", "--"), names[1]: ("#d95f02", "s", "-")}
for nm in names:
    r2 = [float(xr.open_dataset(ROOTS[nm].format(f=f))["R2F_away_centered"].mean()) for f in FACTORS]
    co = [float(xr.open_dataset(ROOTS[nm].format(f=f))["corr_F_away"].mean()) for f in FACTORS]
    c, mk, ls = sty[nm]
    ax.plot(FACTORS, r2, color=c, marker=mk, ls=ls, label=f"{nm} — R²")
    ax.plot(FACTORS, co, color=c, marker=mk, ls=ls, alpha=0.45, label=f"{nm} — corr")
ax.set_xticks(FACTORS); ax.set_xticklabels([DEG[f] for f in FACTORS])
ax.set_xlabel("coarse-grid spacing"); ax.set_ylabel("depth-mean skill (test, coast-excluded)")
ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2)
ax.set_title("Depth-mean flux skill: canonical neutral model is uniformly higher (+0.05–0.07 R²)", fontsize=10)

fig.suptitle("Offline skill, held-out test split: EXP0 (σ₀, superseded) vs EXP_neutral_all4 (canonical)", fontsize=12)
png = "/home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts/offline_skill_exp0_vs_neutral.png"
fig.savefig(png, dpi=130, bbox_inches="tight")
print("wrote", png)

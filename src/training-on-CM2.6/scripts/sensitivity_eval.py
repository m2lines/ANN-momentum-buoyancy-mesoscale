"""Load + aggregate the sensitivity-sweep skill files for the evaluation notebooks.

Reads /scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity/<name>/skill-test/factor-*.nc
(written by run_sensitivity.py). Kept thin so the per-axis notebooks in
DB_notebooks/evaluate_model_design/ stay readable.
"""
import os
import numpy as np
import xarray as xr

ROOT = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity')
FACTORS = [4, 9, 12, 15]
SPACING = {4: 0.4, 9: 0.9, 12: 1.2, 15: 1.5}   # coarse-grid spacing [deg]


def load_config(name):
    """factor -> skill Dataset for one config, or None if nothing is there yet."""
    out = {}
    for f in FACTORS:
        p = f'{ROOT}/{name}/skill-test/factor-{f}.nc'
        if os.path.exists(p):
            out[f] = xr.open_dataset(p)
    return out or None


def is_done(name):
    return all(os.path.exists(f'{ROOT}/{name}/skill-test/factor-{f}.nc') for f in FACTORS)


def depth_mean(name, metric='R2F'):
    """factor -> depth-mean of a metric for one config (None if missing)."""
    d = load_config(name)
    if d is None:
        return None
    return {f: float(d[f][metric].mean()) for f in d}


def depth_profile(name, factor, metric='R2F'):
    """(depth[m], metric) arrays for one config/factor, for depth-profile plots."""
    d = load_config(name)
    if d is None or factor not in d:
        return None, None
    s = d[factor]
    return s.zl.values, s[metric].values


def table(names, metrics=('R2F', 'corr_F', 'R2F_along', 'R2F_across')):
    """Tidy rows (config, factor, metric, value=depth-mean) across a list of configs."""
    rows = []
    for name in names:
        d = load_config(name)
        if d is None:
            continue
        for f in d:
            for m in metrics:
                if m in d[f]:
                    rows.append(dict(config=name, factor=f, metric=m, value=float(d[f][m].mean())))
    return rows

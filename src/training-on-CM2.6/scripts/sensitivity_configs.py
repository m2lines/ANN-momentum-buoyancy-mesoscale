"""Single source of truth for the offline sensitivity sweep.

See ../SENSITIVITY_STUDY_PLAN.md. Each entry is one trained model, identified by its
(stencil_size, hidden_layers, seed). run_sensitivity.py launches them to
/scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity/<name>/; the evaluation notebooks in
DB_notebooks/evaluate_model_design/ read the resulting skill files. One thing varies at a
time from the canonical baseline (stencil 3x3, hidden [32,32], seed 0 -- i.e. EXP1).

Configs are auto-named `st<stencil>_h<hidden>_s<seed>` so each is self-describing and the
set de-duplicates naturally (the baseline is shared across axes, run once).
"""


def cfg(stencil_size, hidden_layers, seed, rotated=False, loss='mse'):
    h = hidden_layers.strip('[]').replace(', ', '-').replace(',', '-')
    name = (f'st{stencil_size}_h{h}_s{seed}'
            + ('_rot' if rotated else '') + ('' if loss == 'mse' else f'_{loss}'))
    return dict(name=name, stencil_size=stencil_size, hidden_layers=hidden_layers,
                seed=seed, rotated=int(rotated), loss=loss)


# Phase 1 -- no new code needed (only stencil / hidden / seed vary).
_phase1 = [
    cfg(3, '[32,32]', 0),                                          # canonical baseline (= EXP1)
    # impact_stencil: lateral non-locality (1x1, 3x3*, 5x5)
    cfg(1, '[32,32]', 0), cfg(5, '[32,32]', 0),
    # impact_model_sizes: width sweep, 3 seeds each (Part 1's most sensitive axis)
    *[cfg(3, w, s) for w in ('[16,16]', '[48,48]', '[64,64]') for s in (0, 1, 2)],
    cfg(3, '[32,32]', 1), cfg(3, '[32,32]', 2),                    # baseline width, extra seeds
    # impact_model_sizes: depth (1 vs 2* vs 3 hidden layers)
    cfg(3, '[32]', 0), cfg(3, '[32,32,32]', 0),
    # impact_random_seed: init/sampling noise at the baseline config (seed 0 = baseline)
    cfg(3, '[32,32]', 3), cfg(3, '[32,32]', 4),
]

# Phase 2 -- impact_rotation: continuous flow-aligned rotation (Part 1). Baseline (off) vs on,
# at the canonical config. The on-the-fly rotation in ANN_rho_inference is verified equivariant
# (scripts/test_rotation_equivariance.py).
# impact_loss_style: training loss MSE (baseline) vs MAE (Part 1).
_phase2 = [cfg(3, '[32,32]', 0, rotated=True),
           cfg(3, '[32,32]', 0, loss='mae')]

# de-duplicate (the baseline is shared by several axes)
CONFIGS = {c['name']: c for c in _phase1 + _phase2}

# Phase 3+ (still need new code paths -- non-dim off, extra inputs, loss style).

# -- selectors the evaluation notebooks use to pull each axis's subset --
BASELINE = cfg(3, '[32,32]', 0)['name']

def stencil_axis():       # impact_stencil
    return [cfg(s, '[32,32]', 0)['name'] for s in (1, 3, 5)]

def model_size_axis():    # impact_model_sizes (widths at seed 0, plus depth)
    return [cfg(3, w, 0)['name'] for w in ('[16,16]', '[32,32]', '[48,48]', '[64,64]')] \
         + [cfg(3, '[32]', 0)['name'], cfg(3, '[32,32,32]', 0)['name']]

def model_size_seeds(width='[32,32]'):  # seed spread at a given width (for error bars)
    return [cfg(3, width, s)['name'] for s in (0, 1, 2)]

def seed_axis():          # impact_random_seed (baseline config across seeds)
    return [cfg(3, '[32,32]', s)['name'] for s in range(5)]

def rotation_axis():      # impact_rotation (flow-aligned off vs on)
    return [cfg(3, '[32,32]', 0)['name'], cfg(3, '[32,32]', 0, rotated=True)['name']]

def loss_style_axis():    # impact_loss_style (mse vs mae)
    return [cfg(3, '[32,32]', 0)['name'], cfg(3, '[32,32]', 0, loss='mae')['name']]

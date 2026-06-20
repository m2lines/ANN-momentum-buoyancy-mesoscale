# Offline sensitivity study — plan (Part 2, CM2.6 3-D buoyancy fluxes)

Replicate the offline design-sensitivity study of Part 1 (Balwada et al., `papers/Balwada_etal2026/`)
on the new dataset: coarse-grained CM2.6, 3-D, density (ρ) fluxes. Goals:
1. Justify / refine the canonical config for the buoyancy model.
2. Resolve the open §3.1 decisions — capacity (`[32,32]` enough?), the rotation/symmetry
   simplification, non-dimensionalization — empirically, as Part 1 did.
3. Produce the sensitivity figures for §3.3, mirroring Part 1's figure set.

Scope (agreed 2026-06-19): **full Part-1 replication** — all axes; all decisions in play.

## Conventions (agreed 2026-06-19, mirroring Part 1's repo layout)
- **Compute in `scripts/`** (heavy, slurm): `sensitivity_configs.py` is the single source of
  truth (the config list); `run_sensitivity.py` is the regenerate-script that launches one
  train+skill slurm job per config. (Part 1 trained in notebooks; here training is slurm
  because CM2.6 is 130GB/GPU — everything *downstream* of the skill files is notebooks.)
- **Figures in notebooks**, organized like Part 1's `evaluate_model_design/`: one subfolder
  per axis under `DB_notebooks/evaluate_model_design/` (`impact_stencil/`, `impact_model_sizes/`,
  `impact_nondim/`, `impact_rotation/`, `impact_more_inputs/`, `impact_loss_style/`,
  `impact_random_seed/`), each with the eval/figure notebook for that axis.
- **Paper figures** in `DB_notebooks/paper_figure_notebooks/`: minimal main-text offline figure
  + appendix `FigureH*` sensitivity figures, one notebook per figure (Part 1 pattern).
- **Shared code** stays in the existing `helpers/` (the Part-2 analog of Part 1's `modules/`).
- **Artifacts: scratch-only + regenerate**. Models + skill netcdfs live at
  `/scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity/<name>/`; `run_sensitivity.py`
  regenerates them. Nothing large committed; notebooks read from scratch.
- Human-readable throughout (Pavel's style): config list reads like a table, launcher is a
  loop, notebooks are markdown + plot cells.

Enabling context: training is now ~6 min/run on GPU and skill eval is batched (~minutes),
so a 30–40-run sweep is a few GPU-hours, not days. `--seed` gives reproducibility. The
float64 fix (`SGS_skill_rho`) makes the along/across metrics trustworthy.

## Canonical baseline
stencil 3×3 · hidden `[32,32]` · non-dim ON · rotation/symmetry OFF · 5 features
(rhox, rhoy, sh_xx, sh_xy, rel_vort) · per-slice normalized MSE · factors [4,9,12,15] ·
500 iters · seed 0 · device cuda. (This is EXP1/EXP2.) Every axis varies ONE thing from this.

## Sweep axes (one-at-a-time)

| # | Axis | Configs | Part 1 analog | Needs |
|---|---|---|---|---|
| 1 | **Stencil** | 1×1, 3×3*, 5×5, (7×7) | headline; 1→3 big, 3→5 small | nothing (data has the fields; `--stencil_size`) |
| 2 | **Capacity (width)** | `[16,16]`, `[32,32]`*, `[48,48]`, `[64,64]` | "most sensitive hyperparameter" | nothing (`--hidden_layers`) |
| 2b| **Capacity (depth)** | `[32]`, `[32,32]`*, `[32,32,32]` | same | nothing |
| 3 | **Non-dim / range-limiting** | ON*, OFF | "most important generalization feature" | a no-norm code path in ANN_rho_inference/train |
| 4 | **Rotation + symmetry** | OFF*, ON | the open §3.1 call | port Part 1's flow-aligned frame + symmetry aug to the ρ pipeline |
| 5 | **Input features** | 5* ; −vorticity ; +N²/strat ; +divergence | "minimal set; future inputs" | feature-set plumbing in feature build |
| 6 | **Loss style** | norm-MSE* ; MAE ; unnormalized | their loss-style study | a loss option in train_rho_fluxes |

(* = baseline)

## Axes that come for free (no extra runs — already in data/eval)
- **Resolution / filter scale** — factors 4/9/12/15; every config yields skill-vs-Δ.
- **Depth** — skill-vs-depth; the genuinely new Part-2 axis (Part 1 was 2-layer).
- **Along/across decomposition** — a diagnostic on every config (now float64-clean).

## Evaluation protocol
- Metrics per config: R²F, corr_F (combined), R²F_along/across — as functions of
  **(resolution, depth)**; report depth-mean and depth-resolved. (Add the `_along_map`/
  `_across_map` spatial variants once we want maps — 2-line addition to SGS_skill_rho.)
- Evaluate on the **test** split (held-out model years 199–200). Tune nothing on test.
- **Seeds**: capacity axis with 2–3 seeds (most sensitive + noisy); single seed elsewhere.
- **Deep-ocean caveat**: report skill over the depth range where the flux is non-negligible,
  or note the known spectral-bias decline at depth (Part 1 line 556 — low signal variance →
  lower offline skill); don't let abyssal levels distort headline numbers.

## Infrastructure to build (Phase 0)
1. Expose `--stencil_size` and `--hidden_layers` (and later `--seed` ensembles) in
   `slurm_train_ann.sh` (currently HIDDEN/ITERS/SEED only — add STENCIL).
2. A **sweep driver** that launches the config list, each to its own `EXP_<tag>` path,
   and runs train + test-skill per config.
3. A **skill-aggregation + plotting** notebook (`DB_notebooks/sensitivity_*.ipynb`) that
   collects all `EXP_<tag>/skill-test/` and makes the comparison figures (skill-vs-stencil,
   skill-vs-capacity, skill-vs-resolution, depth profiles, along/across) — mirroring Part 1.
4. Axes 3–6 need code paths that don't exist yet for the ρ pipeline (non-dim off, rotation,
   feature-set, loss option) — these are the only non-trivial code work.

## Phasing & rough cost
- **Phase 0** — infra + one validated end-to-end run. (~½ day)
- **Phase 1** — stencil + capacity (Part-1 headlines; justifies canonical config; answers
  "is [32,32] enough?"). ~20 runs (capacity × seeds) ≈ a few GPU-hours. *No new code.*
- **Phase 2** — non-dim on/off + rotation/symmetry on/off (the "most important features" +
  the §3.1 call). Gated on the two code paths.
- **Phase 3** — input features + loss style (exploratory). Gated on plumbing.
- **Phase 4** — synthesis figures → §3.3; update PLAN/INVENTORY; decide the canonical config.

## Phase 1 results (2026-06-20) — DONE
18 configs trained + scored on the test split; figures in
`DB_notebooks/evaluate_model_design/{impact_stencil,impact_model_sizes,impact_random_seed}`.
Headline = mean R²F over the 4 resolutions (depth-mean per factor, then averaged):

- **Stencil** (hidden [32,32]): 1×1 = 0.55, **3×3 = 0.70**, 5×5 = 0.73. Big jump 1×1→3×3
  (+0.15), small 3×3→5×5 (+0.03). **3×3 is the sweet spot** — matches Part 1.
- **Capacity** (stencil 3×3, width): [16,16] = 0.66, **[32,32] = 0.70**, [48,48] = 0.72,
  [64,64] = 0.73. Diminishing returns; **[32,32] sits slightly *under* the plateau** —
  [48,48] (Part 1's choice) gives +0.017, [64,64] +0.026, and **test** R² rises (not
  overfitting). Depth: 1 layer = 0.68, 2 = 0.70, 3 = 0.705 (3rd layer adds nothing).
- **Seed** (baseline ×5): mean 0.7029, **std 0.0007** (range 0.002) — negligible, so all the
  differences above are real and well above run-to-run noise.

Decision surfaced for DB: **consider bumping the canonical hidden to [48,48]** (matches Part 1,
+~0.017 R² at negligible cost) vs keeping [32,32]. Stencil 3×3 confirmed; 5×5 optional (+0.03).

## Outputs
- A figure set mirroring Part 1: R²/corr vs stencil, vs capacity, vs resolution; depth
  profiles; along/across maps/profiles — for §3.3.
- A decision on the canonical config (stencil, capacity, rotation, non-dim) with evidence.
- Resolution of the §3.1 rotation/symmetry open item.

## Open risks / decisions
- Phase 2–3 require new ρ-pipeline code (non-dim off, rotation, features, loss) — non-trivial;
  the rotation path especially (port from Part 1's thickness-flux code).
- Slow-node I/O on the preload can make individual jobs ~30 min; parallelize across GPU jobs.
- Capacity vs optimizer: we already showed (2026-06-19) the offline ceiling is data-limited,
  not optimization-limited; capacity is the remaining model-side lever, bounded by Part 1's
  observed skill-vs-params plateau.

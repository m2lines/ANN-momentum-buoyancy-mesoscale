# Part 2 — Buoyancy flux parameterization with realistic grids

Working plan for extending the data-driven mesoscale eddy-induced transport parameterization from idealized 2-layer configurations (Part 1, Balwada et al. 2026) to realistic 3D stratification using CM2.6 training data. The main change from Part 1 is that the target quantity shifts from thickness fluxes to ρ-fluxes, which are the natural 3D analog and directly relevant for ocean circulation. The training pipeline will reuse much of Pavel Perezhogin's pipeline from Perezhogin et al. 2025.

## Goal

**Part 2** of the Balwada et al. two-paper series on ANN-based parameterizations of mesoscale eddy-induced transport. Part 1 establishes the method in 2-layer Phillips and Double-Gyre configurations. Part 2 tests whether the same parameterization structure generalizes to realistic 3D stratification when trained on CM2.6 data.

- **Target quantity**: subfilter ρ-fluxes (Fx, Fy) — 3D/arbitrary vertical grid analog of Part 1's thickness fluxes
- **Training data**: CM2.6 at coarsening factors 4, 9, 12, 15 (reuse Pavel's pipeline)
- **Online platform**: MOM6, via the `rho_flux_ANN_gfdl_ready` branch 
- **Co-authors**: TBD; Balwada lead, Perezhogin co-dev

## Project stance

The heavy lifting — implementation, training, online runs, analysis — has mostly been done. What remains:

1. **Clean up** the codebase and analysis so the story is legible and reproducible
2. **Check the code carefully** for correctness before trusting results
3. **Write the paper**

This plan is organized around those three goals, not around building new capabilities from scratch.

## What already exists (to be inventoried)

Rough catalog — to be refined in M1:

- **Python training pipeline**: buoyancy extensions of Pavel's pipeline in `src/training-on-CM2.6/`. Known issues listed below.
- **Trained ANN artifacts**: **CANONICAL MODEL IDENTIFIED (2026-06-16)** — `CM26_ML_models/ocean3d/subfilter/FGR3/buoyancy/hidden-layer-32-32/seed-default/model/ann_instance_20Dec.nc` (committed in `3df5f40`). Config: stencil 3, hidden `[32,32]`, FGR3, factors `[4,9,12,15]`, depth `arange(0,50,2)`, 500 iters, `path_save=EXP0`. This is the `ann_instance_20Dec.nc` that `offline_eval.ipynb` loads (from `scratch/.../FGR3/EXP0/model/`). The `training_experiments/*_6Nov.nc` artifacts are earlier/superseded, NOT canonical.
- **Online MOM6 implementation**: `rho_flux_ANN_gfdl_ready` branch in MOM6 — Fortran code already "gfdl_ready". See `/home/db194/MOM6-examples/src/MOM6/src/parameterizations/lateral/MOM_meso_sfn_ANN.F90`.
- **Analysis notebooks** in `DB_notebooks/`:
  - `NW2_analysis/` — 5 notebooks.
  - `DG_analysis/` — 1 notebook.
  - `config_checks/` — 4 notebooks (doublegyre, neverworld2, OM4, trained logger).
  - Top-level: `PE_impacts`, `relative_mags_forcing`, `offline_eval`, `11-Dhruv-Nov-runs`.
  - **Channel-model analyses moved out 2026-05-18** to sibling repo `../ANN-channel-forcing-sensitivity/` — wind-stress / cb / cu sweep is now a standalone project. This repo covers training + cross-config validation only.
- **Simulation output**: on disk in `/vast/` or `/scratch/` — locations to be documented.

## Known code issues (from 2026-04-21 audit)

Uncommitted / in-flight:
- `src/training-on-CM2.6/scripts/train_script_rho_fluxes.py` (entry point, has bugs)
- `src/training-on-CM2.6/scripts/slurm_train_ann.sh`
- ~~`predict_ANN_rho()` stub in `cm26.py`~~ → **IMPLEMENTED (2026-06-16)**: whole-dataset ρ inference (loops `ANN_rho_inference` over time/zl → Fx/Fy + Fx_pred/Fy_pred). Added `SGS_skill_rho()` (R²F/corr_F) alongside it.
- `README.md` updates (fork blurb + buoyancy section)

Bugs:
- `cm26.py:118` rawdata path hardcoded to `/vast/pp2681/` (Pavel's user) — portability
- ~~`train_script_rho_fluxes.py` references `args.gradient_features`~~ → **FIXED (2026-06-16)**: testing block now calls `predict_ANN_rho(ann_Tall, stencil_size=...).SGS_skill_rho()`, no `gradient_features`.
- ~~`train_script_rho_fluxes.py` testing block calls momentum `predict_ANN`~~ → **FIXED (2026-06-16)**: now uses the real `predict_ANN_rho` + `SGS_skill_rho`.

## Milestone 1 — Inventory and audit  [open]

Before we can trust any number, we need to know (a) what exists, (b) what's correct, and (c) what's canonical.

- [ ] **Inventory analysis notebooks** by subdirectory: tag each as canonical / superseded / scratch
- [x] **Inventory trained models (2026-06-16)**: canonical = `ann_instance_20Dec.nc` (EXP0; stencil 3, hidden `[32,32]`, FGR3, factors `[4,9,12,15]`), committed under `CM26_ML_models/.../FGR3/buoyancy/hidden-layer-32-32/seed-default/`; used by `offline_eval.ipynb` and the online MOM6 runs. `training_experiments/*_6Nov` are superseded.
- [ ] **Inventory online simulations**: list each MOM6 run, its config, its output location on disk
- [~] **Code review — training pipeline** (`add_subfilter_forcing_rho`, `ANN_rho_inference`, `train_rho_fluxes.py`): check math, compare against Pavel's momentum equivalents
  - [x] **Train↔online "implicit contract" verified (2026-06-16)**: input features (5 channels `rhox, rhoy, sh_xx, sh_xy, rel_vort`, same order), gradient sign conventions (`sh_xx=dudx-dvdy`, `sh_xy=dvdx+dudy`, `vort=dvdx-dudy`), Frobenius-norm normalization, and dimensional prefactor (`rho_norm·gradv_norm·Δ²`) all MATCH between `ANN_rho_inference` (`state_functions.py:1574`) and `meso_sfn_ANN_compute`. Both intentionally drop horizontal divergence.
  - [x] **Flux sign verified**: training target `Fx = ρ̄ū − ρu‾ = −u'ρ'‾` (`cm26.py:585`); Fortran comment "ANN outputs −u'rho', so we negate" (`F90:285`) correctly reconstructs `+u'ρ'‾`. Consistent, no sign bug.
  - [x] Residual closed (2026-06-16): `compute_features` builds `sh_xy_h = interp(dvdx+dudy)`, `rel_vort_h = interp(dvdx-dudy)` — signs match the Fortran inputs.
  - [x] **Full math/sign audit (2026-06-16)**: `add_subfilter_forcing_rho` computes `F = ρ̄ū - ρu‾ = -u'ρ'`, mirroring Pavel's momentum `T = ū·ū - uu‾` (it was lifted from the commented-out Fx/Fy in `compute_subfilter_forcing`). Offline internally consistent: both the flux target and the `rhox/rhoy` inputs (`vertical_shear_geostrophic`) derive from `state.rho()` = potential density σ0. Per-slice `F_norm` is only a loss weighting, not baked into the exported model. **No core-math bugs.**
  - [x] **BUG FIXED (2026-06-16) — Polar-Fold rows leak into ρ training**: added `drop_polar_fold()` in `train_rho_fluxes.py` (drops top 2 `yh`+`yq` rows, reconstructs `DatasetCM26` so data/param/state stay aligned), applied at both train and validate `select2d` calls. Mirrors `fetch_data`'s `isel(yh=slice(None,-2))`. Syntax-checked; not yet run end-to-end (needs data/env).
  - [x] Minor doc fix (2026-06-16): corrected `vertical_shear_geostrophic` docstring — `self.data.rho` is potential σ0, not in-situ.
- [~] **Code review — MOM6 Fortran** (`MOM_meso_sfn_ANN.F90`): verify inputs/outputs, sign conventions, boundary handling, clamping
  - [x] inputs/outputs + sign conventions cross-checked vs training (see above).
  - [ ] still to audit: boundary handling (`min_dist_from_boundary`, ψ=0 at surface/bottom via interior-only fill), clamping (`flux_clamp`, `Upsilon_clamp`, `mag_grad_floor`), Ferrari-2010 streamfunction `Υ = F_h/|∇₃ρ|`.
- [x] **NEW audit item — density-gradient "kind" consistency → CONFIRMED INCONSISTENCY (2026-06-16)**: training builds `rhox/rhoy` from **surface-referenced potential density σ₀** (`rho()`=`gsw.sigma0`, ref p=0; gradient at constant model level — `state_functions.py:1776,1948`). Online, `calc_isoneutral_slopes` evaluates density derivs **at the local interface pressure** (`pres_u`, `MOM_isopycnal_slopes.F90:312,336`) → ∇ρ in the neutral/locally-referenced framework. These agree near surface but **diverge with depth** (thermobaricity), in both magnitude and direction; training spans full column (`depth_idx=arange(0,50,2)`), so a real distributional shift that `C_ANN` (a scalar) cannot correct. **Likely contributor to imperfect online skill.**
  - Resolution options: (1) retrain on locally-referenced density gradients to match `calc_isoneutral_slopes` [cleanest]; (2) add a σ₀-gradient path to the Fortran; (3) diagnose ∇σ₀ vs ∇ρ_local in CM2.6 and argue negligible [risky].
  - Secondary, unchased: training differences at constant-z (CM2.6 z-coord) vs online at constant-interface-K (z* in OM4) — not identical in general vertical coordinate.
- [x] **Offline-skill code reuse map (deep audit 2026-06-16)** — what already exists, so we don't reinvent (Pavel built most of it; 3-agent audit, all agree there is *zero* existing R²/corr for Fx/Fy):
  - **`M2()`** (`cm26.py:742`, nested in `SGS_skill`) — generic 2nd-moment (centering/masking/dim-reduction + `M2u`/`M2v` grid variants). Already works for any field incl. Fx/Fy. Building block for all R²/corr.
  - **`SGS_skill()`** (`cm26.py:719-967`) — full momentum suite: R²T + corr_T (map/centered/away/lon variants, lines 789-828), SGS dissipation, transfer/power spectra (NA/Pacific/Equator/ACC), dEdt. **Momentum-only.** The R²T/corr_T block is the template for a ρ R²F/corr_F.
  - **`predict_ANN()`** (`cm26.py:613-649`) — working whole-dataset momentum prediction (loops time/zl → xarray). Structural template for `predict_ANN_rho`.
  - **`ANN_rho_inference()`** (`state_functions.py:1574`) — the only working ρ predictor; per-2D-slice; `return_xarray=True` gives Fx/Fy for comparison-to-truth.
  - **`predict_ANN_rho()`** (`cm26.py:651`) — ✅ now implemented (2026-06-16), plus `SGS_skill_rho()` (R²F/corr_F).
  - **`notebooks/Figure-1.ipynb`** (Pavel) — R²/corr heatmaps across factors×depths via `SGS_skill`; the plotting template for §3.3 ρ skill.
  - Qualitative ρ (exists): `offline_eval.ipynb`, `training_experiments/train_ann_rho_test.ipynb` — `ANN_rho_inference` vs CM2.6 truth (maps/histograms, no R²).
  - **Status (2026-06-19)**: (1) `predict_ANN_rho` ✅ done; (2) R²F/corr_F via `SGS_skill_rho` ✅ done; (3) **offline R²/corr numbers PRODUCED (2026-06-19)** — see "Offline skill results" below. Driver: `scripts/compute_skill_rho.py` (+ `slurm_skill_rho.sh`), figures+playground in `DB_notebooks/offline_skill_rho.ipynb`. Remaining: final Figure-3.3 panel styling (Pavel Figure-1 template).

- [x] **Offline skill results (2026-06-19)** — canonical ANN `ann_instance_20Dec.nc` scored on regenerated CM2.6, depth-mean R²F (combined Fx,Fy) per coarse-grid spacing:

  | Δ (deg) | factor | train | validate | test | corr_F (test) |
  |---|---|---|---|---|---|
  | 0.4 | 4  | 0.793 | 0.798 | 0.80 | 0.88 |
  | 0.9 | 9  | 0.710 | 0.707 | 0.73 | 0.83 |
  | 1.2 | 12 | 0.652 | 0.647 | 0.67 | 0.79 |
  | 1.5 | 15 | 0.591 | 0.584 | 0.61 | 0.75 |

  - **No overfitting**: train ≈ validate ≈ test at every resolution (spread ≤0.03; train is a hair *below*, not above). The reported R² is the parameterization's intrinsic predictive limit at each Δ, not a generalization gap. Monotonic decline with coarsening is a real physics effect (more subfilter variance to explain), identical across splits. Equal-N comparison (24 snapshots each; train evenly subsampled from 96 via `NTIME=24` — `predict_ANN_rho` materializes full fields so 96-snapshot train OOMs at 128GB).
  - **Settles the rotation/symmetry question (open item below)**: strong skill *without* Part-1's flow-aligned rotation or symmetry augmentation → keep the simplification; no retrain needed. Frame as explicit simplification/future-work in the paper.
  - Also computed (in the skill files, not yet in paper): along/across-gradient R² decomposition relative to ∇ₕρ. **Interpretation deliberately open — discuss with Dhruv before asserting physics; "across" is a geometric projection, NOT "rotational".**
  - Skill files: `/scratch/db194/CM26_ML_models/FGR3/EXP0/skill-{train,validate,test}-rho/factor-*.nc`.
- [ ] **Reproducibility spot-check**: rerun one training end-to-end, confirm it matches a canonical trained model
- [~] **Fix known bugs**: ~~`gradient_features` handling~~ ✅, ~~`predict_ANN_rho` stub~~ ✅ (implemented + `SGS_skill_rho`); remaining: `cm26.py` hardcoded `/vast/pp2681/` rawdata path (portability).
- [ ] **Commit queue cleanup**: land README + training script + slurm script + plan + reorg

## Milestone 2 — Cleanup and consolidation  [after M1]

Using the M1 inventory, reduce the analysis sprawl to a canonical story.

- [ ] Delete or archive superseded notebooks (tagged in M1)
- [ ] Consolidate multi-variant analyses into one canonical notebook per scientific claim
- [ ] Extract reusable analysis code from notebooks into helper `.py` modules where it repeats
- [ ] Ensure each canonical notebook runs top-to-bottom without manual intervention
- [ ] Write short READMEs inside each analysis subdir describing what's there

## Milestone 3 — Write the paper  [after M2]

**Paper draft lives in a separate repo**: `github.com/dhruvbalwada/mesoscale_b_ml_parameterization` (Overleaf GitHub Sync, branch `main`). Title: "...mesoscale eddy-induced transport: A hierarchy of configurations". Inline review via `\note[CC]`/`\note[DB]` trackchanges notes.

- [~] Outline Part 2 section structure — current skeleton: §1 Intro, §2 Sub-grid buoyancy fluxes (drafted/reconciled 2026-06-16), §3 ML model design+training+implementation (3.1 design, 3.2 training/CM2.6, 3.3 offline skill, 3.4 implementation-in-MOM6), §4 Evaluation (channel baseline → NW2 → OM4), §5 Discussion, App A Double-Gyre (bridge to Part 1).
  - [x] §2 reconciled: de-dup decomposition, bridge to ML target + Part-1 limit, `b=-gρ/ρ0` defined, notation → superscript `F^b`.
  - [ ] **TODO appendix** `app:flux_decomposition`: honest "what the skew-only closure discards" (diapycnal residual, rotational-flux gauge freedom, QG-limit). Referenced from §2 but unwritten.
- [ ] Draft figures list and map each to the canonical notebook producing it
- [ ] Draft sections iteratively, in priority order:
  - Introduction (lighter; Part 1 has the big-picture framing)
  - Data and training (what changes from Part 1 / Pavel)
  - Results: offline skill in 3D
  - Results: online experiments (channel, possibly others)
  - Discussion
- [ ] Figure panel production — final versions, consistent styling
- [ ] Internal review with co-authors

## Open questions

- **Architecture**: Does the 2-layer ANN architecture (stencil size, feature set, hidden layers) transfer directly to 3D, or does 3D need different features?
  - **Known simplification (confirmed 2026-06-16)**: the ρ-flux ANN **drops Part 1's flow-aligned coordinate rotation and symmetry augmentation** — inputs are fed in the model grid frame (no rotation; `--symmetries` commented out in the trainer; `ANN_rho_inference` has no rotation args, unlike the momentum `ANN_inference`). Was a move-fast simplification, not principled. ~~**Decision before submission**: (a) keep, or (b) retrain in flow-aligned frame.~~ → **LEANING (a), informed by 2026-06-19 offline skill**: R²F=0.80→0.59 (Δ=0.4°→1.5°) *without* rotation/symmetry, train≈val≈test → the simplification doesn't visibly cost offline skill or generalization. Keep it; frame as explicit simplification/future-work. (Confirm with Dhruv; retain (b) as fallback if online results disappoint.)
- **Canonical online experiment**: which channel-sweep variant is the headline result? (lives in sibling repo `ANN-channel-forcing-sensitivity/` — see its INVENTORY.md)
- ~~**Online scope (this repo)**~~ → **RESOLVED (2026-06-16)**: config hierarchy = channel baseline → NW2 → OM4 (main §4); Double-Gyre in appendix to bridge to Part 1. Channel *sensitivity sweep* stays in sibling repo; only a baseline channel run appears here.
- ~~**Scope**: joint vs buoyancy-only~~ → **RESOLVED (2026-06-16)**: buoyancy-only. Variable convention: hybrid (`b` in theory, `ρ` in implementation, bridged by `b=-gρ/ρ0`).
- **Author list**: Lock in co-authors and expected contributions.

## Log

- **2026-04-21**: repo reorganized (DB_notebooks/ subdirs, README updated); initial plan created. Confirmed scope as Part 2 extension of Balwada et al. 2026.
- **2026-04-21**: reframed plan — project stance is audit / cleanup / write, not build-from-scratch. Replaced "train end-to-end" milestones with inventory + consolidation + paper-writing structure.
- **2026-05-18**: split off `DB_notebooks/channel_model_analysis/` into sibling repo `ANN-channel-forcing-sensitivity/`. This repo now covers training + offline eval + cross-configuration validation; the channel wind-stress sweep is its own project.
- **2026-06-19** (perf pass): optimized the ρ-flux training + skill pipeline — see `src/training-on-CM2.6/PERFORMANCE.md` for the full writeup. Training loop ~46.5→0.70 s/iter (preload + `--validate_every` throttle, ~7.5× bit-exact CPU; then `--device cuda` GPU, ~7.4× same-node; end-to-end run ~7 h→~42 min). Offline skill: batched `predict_ANN_rho` over time (shared batch-capable `ANN_rho_inference`) → ~10× on GPU, bit-exact. All changes validated bit-identical vs the pre-optimization code (commit `772c23b`, which is how to run the original approach). Corrected two wrong calls in-flight (GPU "hurts" skill — anomalous node; output-assembly vectorization — regressed, reverted).
- **2026-06-19**: produced the offline ρ-flux skill numbers (train/validate/test R²F, table in M1). Clean train≈val≈test → no overfitting; strong skill without flow-aligned rotation → keep that simplification. Recovered from the Greene→torch migration: paths migrated to `/scratch`, full CM2.6 dataset regenerated (all 4 factors, 786GB). Fixed a quota-crash artifact (`factor-15/train-71.nc` written all-NaN; was the only corrupt file) and hardened `generate_3d_datasets.py` resume to validate coords before skip-if-exists. Added `NTIME` subsampling to `compute_skill_rho.py` (equal-N split comparison, avoids OOM).
- **2026-06-16**: verified the train↔online "implicit contract" for the ρ-flux pipeline — input features, gradient sign conventions, normalization, dimensional prefactor, and flux sign all match between `ANN_rho_inference` and `MOM_meso_sfn_ANN.F90`. No wiring/sign bugs on the two points that would have invalidated online results. Surfaced one deeper open item: potential-density (offline) vs isoneutral-slope (online) density-gradient consistency. Began Part 2 paper planning: scope = buoyancy-only; online reality not yet settled (expect pros+cons) → paper will be method-forward with offline CM2.6 skill load-bearing and online NW2/OM4 framed as first realistic tests.
- **2026-06-16**: started writing in the paper repo `dhruvbalwada/mesoscale_b_ml_parameterization` (Overleaf-synced). Reconciled §2 (Dhruv had a fuller draft than expected — skew/streamfunction decomposition already present); de-duplicated, added ML-target + Part-1-limit bridge, defined `b=-gρ/ρ0`, standardized notation to `F^b`. Locked decisions: buoyancy-only; hybrid `b`/`ρ`; config hierarchy channel→NW2→OM4 + DG appendix. Drafting follows Dhruv's Part 1 voice. Pushed to paper repo `main`. Open inline `\note[CC]` threads await Dhruv's reply.

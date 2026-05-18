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
- **Trained ANN artifacts**: `DB_notebooks/training_experiments/` (`ann_instance_6Nov.nc`, `logger_6Nov.nc`, `rho_flux_ann_params.nc`, training notebooks). Canonical version not yet identified.
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
- `predict_ANN_rho()` stub in `cm26.py` (whole-dataset inference, currently a no-op)
- `README.md` updates (fork blurb + buoyancy section)

Bugs:
- `cm26.py:118` rawdata path hardcoded to `/vast/pp2681/` (Pavel's user) — portability
- `train_script_rho_fluxes.py` references `args.gradient_features` that was commented out at parse → AttributeError in testing block
- `train_script_rho_fluxes.py` testing block calls momentum `predict_ANN` rather than a rho predict (because the rho one is a stub)

## Milestone 1 — Inventory and audit  [open]

Before we can trust any number, we need to know (a) what exists, (b) what's correct, and (c) what's canonical.

- [ ] **Inventory analysis notebooks** by subdirectory: tag each as canonical / superseded / scratch
- [ ] **Inventory trained models**: list each trained ANN, its training config, which experiments used it
- [ ] **Inventory online simulations**: list each MOM6 run, its config, its output location on disk
- [ ] **Code review — training pipeline** (`add_subfilter_forcing_rho`, `ANN_rho_inference`, `train_rho_fluxes.py`): check math, compare against Pavel's momentum equivalents
- [ ] **Code review — MOM6 Fortran** (`MOM_meso_sfn_ANN.F90`): verify inputs/outputs, sign conventions, boundary handling, clamping
- [ ] **Reproducibility spot-check**: rerun one training end-to-end, confirm it matches a canonical trained model
- [ ] **Fix known bugs**: cm26.py path, `gradient_features` handling, decide on `predict_ANN_rho` stub
- [ ] **Commit queue cleanup**: land README + training script + slurm script + plan + reorg

## Milestone 2 — Cleanup and consolidation  [after M1]

Using the M1 inventory, reduce the analysis sprawl to a canonical story.

- [ ] Delete or archive superseded notebooks (tagged in M1)
- [ ] Consolidate multi-variant analyses into one canonical notebook per scientific claim
- [ ] Extract reusable analysis code from notebooks into helper `.py` modules where it repeats
- [ ] Ensure each canonical notebook runs top-to-bottom without manual intervention
- [ ] Write short READMEs inside each analysis subdir describing what's there

## Milestone 3 — Write the paper  [after M2]

- [ ] Outline Part 2 section structure (likely parallel to Part 1 §3–§5)
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
- **Canonical online experiment**: which channel-sweep variant is the headline result? (lives in sibling repo `ANN-channel-forcing-sensitivity/` — see its INVENTORY.md)
- **Online scope (this repo)**: NW2 3D / OM4 / double-gyre 3D — what's in scope for cross-config validation in Part 2?
- **Scope**: Joint momentum+buoyancy training vs. buoyancy-only for Part 2?
- **Author list**: Lock in co-authors and expected contributions.

## Log

- **2026-04-21**: repo reorganized (DB_notebooks/ subdirs, README updated); initial plan created. Confirmed scope as Part 2 extension of Balwada et al. 2026.
- **2026-04-21**: reframed plan — project stance is audit / cleanup / write, not build-from-scratch. Replaced "train end-to-end" milestones with inventory + consolidation + paper-writing structure.
- **2026-05-18**: split off `DB_notebooks/channel_model_analysis/` into sibling repo `ANN-channel-forcing-sensitivity/`. This repo now covers training + offline eval + cross-configuration validation; the channel wind-stress sweep is its own project.

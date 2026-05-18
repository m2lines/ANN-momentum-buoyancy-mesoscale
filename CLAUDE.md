# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of [m2lines/ANN-momentum-mesoscale](https://github.com/m2lines/ANN-momentum-mesoscale) (Perezhogin, Zanna, Adcroft — momentum-flux ANN parameterization for MOM6). This fork extends the training pipeline from **momentum fluxes to buoyancy (ρ) fluxes** for Part 2 of a two-paper series (Balwada et al., 3D extension of the 2-layer Phillips / Double-Gyre results in Part 1). See `PLAN.md` for project stance and milestones — the work is in audit/cleanup/write-paper mode, not build-from-scratch.

Two largely independent layers live here:

- **Upstream momentum work (unchanged)** — paper figures (`notebooks/Figure-*.ipynb`), MOM6 implementation (`src/MOM6` submodule, `src/mom6_modifications.patch`), trained momentum ANNs in `CM26_ML_models/ocean3d/subfilter/FGR3/`, online configs in `configurations/{NeverWorld2,OM4}/`. Don't modify unless explicitly asked.
- **Buoyancy extension (this fork's work)** — `src/training-on-CM2.6/helpers/train_rho_fluxes.py`, `scripts/train_script_rho_fluxes.py`, ρ-flux diagnostics added to `helpers/cm26.py`, and all `DB_notebooks/` analyses.

## Authoritative project docs (read these first)

- **`PLAN.md`** — current goals, milestones, known bugs, open questions. Project state of record.
- **`DB_notebooks/INVENTORY.md`** — every analysis notebook classified `canonical` / `superseded` / `scratch` / `unclear`. Consult before editing or referencing a `DB_notebooks/*.ipynb`.
- **`README.md`** — upstream + buoyancy-fork install and run instructions.

The **channel-model wind-stress sweep** lives in a sibling repo, `../ANN-channel-forcing-sensitivity/` (split off 2026-05-18). This repo handles training + offline eval + cross-configuration validation; the channel sensitivity study is its own project.

## Common commands

ρ-flux training (this fork's primary entry point):
```bash
cd src/training-on-CM2.6/scripts/
python train_script_rho_fluxes.py                       # local
sbatch slurm_train_ann.sh                               # cluster (singularity + Pavel_container)
```

Original momentum-flux training (upstream, unchanged):
```bash
cd src/training-on-CM2.6/scripts/
python train_script.py
```

Coarse-grained dataset generation (one factor at a time):
```bash
cd src/training-on-CM2.6/scripts/
python generate_3d_datasets.py --factor=4               # also 9, 12, 15
# Slurm wrappers: slurm_generate_datasets_{4,9,12,15}.sh
```

There is no test suite, lint config, or build step — this is a research codebase. Validation happens by running training/analysis end-to-end and checking outputs.

## Architecture: training pipeline

The pipeline operates on coarse-grained CM2.6 ocean simulation data and trains a per-point ANN to predict subfilter fluxes given local state.

Data flow:
1. **Raw CM2.6** (GFDL hosted) → `scripts/download_raw_data.py` writes to `rawdata` path.
2. **Coarsen + filter** → `scripts/generate_3d_datasets.py --factor=N` produces `train*.nc / test*.nc / validate*.nc / param.nc / permanent_features.nc` per coarsening factor.
3. **Read** → `helpers.cm26.read_datasets()` returns a dict of `DatasetCM26` instances keyed `'{split}-{factor}'`.
4. **Train** → `helpers.train_rho_fluxes.train_ANN_rho_fluxes()` (ρ pipeline) or `helpers.train_ann_fluxes.train_ANN_fluxes()` (momentum pipeline).
5. **Export** → `helpers.ann_tools.export_ANN()` writes a netcdf (`layer_sizes`, `An`/`bn`, input/output norms) consumable by MOM6's `MOM_Zanna_Bolton.F90` / `MOM_meso_sfn_ANN.F90`. The reference for this netcdf format is `DB_notebooks/training_experiments/generate_test_ann_params.ipynb`.

Key modules in `src/training-on-CM2.6/helpers/`:
- `cm26.py` — `DatasetCM26` class, dataset reader, ρ-flux subfilter diagnostics
- `state_functions.py` — `StateFunctions`: local non-dimensionalization, the "dimensional scaling" that makes the ANN generalize across regions/resolutions
- `feature_extractors.py` — stencil-based feature construction from coarse state
- `ann_tools.py` — `ANN` class, `tensor_from_xarray`, `export_ANN`
- `train_ann.py`, `train_ann_fluxes.py`, `train_rho_fluxes.py` — training loops (momentum legacy, momentum fluxes, ρ fluxes respectively)
- `operators.py`, `selectors.py`, `plot_helpers.py` — xgcm-based coarsen/filter operators, region selectors, plotting

## Architecture: online MOM6

`src/MOM6/` is a git submodule pinned to a specific upstream commit with the ANN parameterization patched in. `src/mom6_modifications.patch` is the canonical diff against `e63a8220e` and exists so the modifications survive on Zenodo. See `src/README.md` for the patch-application recipe.

For the **ρ-flux online code**, the Fortran lives on the `rho_flux_ANN_gfdl_ready` branch of Dhruv's MOM6 fork (`MOM_meso_sfn_ANN.F90`), separately from this repo. See `PLAN.md` for status.

## Things to know before changing code

- **Hardcoded user paths**: `helpers/cm26.py` uses `/vast/$USER/CM26_datasets/...` and (for raw data) a hardcoded `/vast/pp2681/` (Pavel's username). `PLAN.md` flags this as a known portability bug; fix only when explicitly working on it.
- **`train_script_rho_fluxes.py` has known bugs** (see `PLAN.md`): the `--gradient_features` and `--symmetries` argparse lines are commented out but referenced later via `args.gradient_features`; the testing block calls the momentum `predict_ANN` because the ρ predictor (`predict_ANN_rho` in `cm26.py`) is still a stub.
- **`DB_notebooks/training_experiments/`** holds in-flight ANN artifacts (`ann_instance_6Nov.nc`, `logger_6Nov.nc`, `rho_flux_ann_params.nc`). The canonical trained model has not yet been identified — don't assume any specific `.nc` here is "the" model without checking `PLAN.md` / asking.
- **Simulation output is off-tree** on `/vast/$USER/` and `/scratch/$USER/`. Notebook savefig/output paths frequently point to Pavel's tree (`/home/pp2681/...`) and need rewriting when re-run locally.
- **Notebook hygiene**: before editing any `DB_notebooks/*.ipynb`, check its tag in `INVENTORY.md`. Channel-model analyses are no longer in this tree — see the sibling `ANN-channel-forcing-sensitivity/` repo.

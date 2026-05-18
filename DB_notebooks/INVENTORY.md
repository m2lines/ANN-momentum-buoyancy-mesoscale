# DB_notebooks Inventory

Audit of analysis notebooks under `DB_notebooks/` to support the Part 2 paper
(3D CM2.6 extension). Each entry gets a tag (`canonical`, `superseded`,
`scratch`, `unclear`) so we can decide what to consolidate into paper figures
versus archive. Classifications are best-effort from titles, first markdown/
code cells, savefig/to_netcdf output paths, and file mtimes — not from
executing notebooks. Flag anything in "Questions for Dhruv" rather than trust.

Mtimes were captured on 2026-04-21. "Jan/Feb 2026" for NW2; Nov–Dec 2025
for training and early 3D channel experiments.

**Channel-model analyses moved out on 2026-05-18** to the sibling repo
[`ANN-channel-forcing-sensitivity`](../../ANN-channel-forcing-sensitivity/)
— the wind-stress / cb / cu sweep is now a standalone scientific study.
What stays in this repo is training, offline evaluation, and
cross-configuration validation (NW2, OM4, Double-Gyre).

---

## Top-level `DB_notebooks/*.ipynb`

- **`PE_impacts.ipynb`** (Nov 6)
  - Purpose: plots of PE / eddy-forcing magnitudes from coarse-grained CM2.6
    subfilter data (`/vast/db194/CM26_datasets/ocean3d/subfilter/FGR3/factor-*`).
  - Tag: `unclear` — could become a paper figure (relative magnitudes of
    forcing terms is paper-relevant), but no markdown, no saved outputs, and
    overlaps substantially with `relative_mags_forcing.ipynb`.
  - Key outputs: none saved.
  - Related: `relative_mags_forcing.ipynb`.
  - Concerns: no title cell; no saved figures; likely superseded by
    `relative_mags_forcing`.

- **`relative_mags_forcing.ipynb`** (Nov 6)
  - Purpose: "Plots of different components" — decomposes the PV/momentum
    forcing (Eden et al. 2023 framework, S^u, S^v) from coarse-grained CM2.6
    and compares relative magnitudes of Reynolds-stress vs b'-flux terms.
  - Tag: `canonical` (candidate) — framed around the exact equations the paper
    introduces; the most likely source of a "why streamfunction/ANN" framing
    figure.
  - Key outputs: none saved via `savefig`; figures are inline.
  - Related: `PE_impacts.ipynb` (likely earlier/overlapping).
  - Concerns: no savefig means figures exist only in notebook cell output; if
    this becomes a paper figure it should save to disk.

- **`offline_eval.ipynb`** (Jan 6)
  - Purpose: offline evaluation of a trained ANN
    (`ann_instance_20Dec.nc`, FGR3/EXP0) against coarse-grained CM2.6 at
    factors [4,9,12,15]. Uses `helpers.train_rho_fluxes`, `helpers.ann_tools`.
  - Tag: `canonical` (candidate) — standard "offline R^2 / scatter" figure
    every ML-param paper has.
  - Key outputs: inline only.
  - Related: `training_experiments/train_ann_rho_test.ipynb` (upstream trainer),
    `config_checks/check_trained_logger.ipynb` (training curves).
  - Concerns: no markdown; need to confirm which ANN instance is "final".

- **`11-Dhruv-Nov-runs.ipynb`** (Jan 6, 6 MB)
  - Purpose: coarse comparison of OM4 runs (Pavel Perezhogin's
    `/scratch/pp2681/mom6/OM4_SIS2` + Dhruv's `/scratch/db194/OM4`) using the
    `CollectionOfExperiments` helper. Plots T, S, MLD, SSH std.
  - Tag: `scratch` — large embedded outputs, exploratory; filename is a date
    marker ("Nov runs"), not a consolidated analysis.
  - Key outputs: inline only.
  - Related: `config_checks/check_OM4.ipynb` (more recent, Apr 20).
  - Concerns: looks superseded by `check_OM4.ipynb`.

---

## `NW2_analysis/` (NeverWorld2)

The `R2`/`R3`/`R4` notebooks are clearly resolution variants (R2=half-degree,
R3=third-degree, R4=quarter-degree).

- **`NW2_analysis_R2.ipynb`** (Feb 6, newest)
  - Purpose: bulk-statistics, speed plots, interface comparisons at R2.
  - Tag: `canonical` (R2 is likely the headline resolution given Feb 6 mtime).
  - Concerns: savefig targets `/home/pp2681/…` (Pavel's tree) — paths stale
    on this machine.

- **`NW2_analysis_R3.ipynb`** (Jan 7)
  - Purpose: R3 variant.
  - Tag: `canonical` (companion resolution).

- **`NW2_analysis_R4.ipynb`** (Jan 7)
  - Purpose: R4 variant.
  - Tag: `canonical` (companion resolution).

- **`NW2_bias_interface_plots.ipynb`** (Feb 7)
  - Purpose: HR (R32) vs LR (R3-stable) longmean + snapshot interface bias
    plots; uses `helpers.plot_helpers` from `src/training-on-CM2.6`.
  - Tag: `canonical` (candidate) — focused bias-figure notebook, newest.

- **`Figure-S3-and-Tables-S2-S3_DB.ipynb`** (Jan 7)
  - Purpose: explicit figure/table reproduction — name indicates this
    corresponds to Supplementary Figure S3 and Tables S2–S3 (of Part 1
    Perezhogin et al. paper, or an earlier draft of Part 2).
  - Tag: `canonical` — clearly paper-figure-generating.
  - Concerns: savefig paths under `/home/pp2681/…` — won't write locally
    without rewriting paths.

---

## `DG_analysis/` (Double-Gyre)

- **`DG_compare_ANNC_20km.ipynb`** (Jan 6)
  - Purpose: 6-cell DG comparison with/without ANN at 20 km; single markdown
    "## DG".
  - Tag: `unclear` — too thin to judge; may be the only DG analysis (could
    be canonical by virtue of being the only one) or a throwaway probe.

---

## `config_checks/` (sanity-check notebooks)

- **`check_OM4.ipynb`** (Apr 20, 5.9 MB — newest in entire tree)
  - Purpose: "OM4 simulation with ANN" — bulk stats, ΔKE vs ΔAPE plots,
    simulated fields.
  - Tag: `canonical` — most recent work; OM4+ANN is the realism endpoint of
    the paper.
  - Concerns: very large notebook; likely embedded outputs; may overlap with
    `11-Dhruv-Nov-runs.ipynb`.

- **`check_doublegyre.ipynb`** (Jan 6)
  - Purpose: KE/APE time series, ANN vs ANN+boundary-filter vs no-ANN for
    `dec2025/DoubleGyre`.
  - Tag: `scratch` (sanity).
  - Related: `DG_analysis/DG_compare_ANNC_20km.ipynb`.

- **`check_neverworld2.ipynb`** (Jan 1)
  - Purpose: "Check vhGM" — diagnostic of GM thickness flux in NW2.
  - Tag: `scratch`.

- **`check_trained_logger.ipynb`** (Dec 21)
  - Purpose: plots a trained-model logger.nc (CM26/FGR3/EXP0 trainer loss
    curves).
  - Tag: `scratch` — routine training-curve check.

---

## `training_experiments/`

- **`train_ann_rho_test.ipynb`** (Dec 22)
  - Purpose: "Training section" + prediction checking. Writes
    `logger_6Nov.nc`.
  - Tag: `unclear` — may be the canonical training script for the ANN
    currently used in runs, or may be a test. Depends on whether training
    has moved into a `src/` module.
  - Key outputs: `logger_6Nov.nc` (present locally, Nov 6).

- **`generate_test_ann_params.ipynb`** (Nov 17, well-documented)
  - Purpose: documents + generates the MOM6 ANN parameter netcdf format
    (layer_sizes, An/bn, input/output norms). Writes
    `./rho_flux_ann_params.nc` and `Phillips_2layer_{20km,}_ANN/ann_params.nc`.
  - Tag: `canonical` (reference/utility) — the definitive how-to for
    producing ANN_PARAMS_FILE; likely referenced in paper methods or SI.

- **`debug_test.ipynb`** (Nov 17)
  - Purpose: debug cells around `helpers.cm26.DatasetCM26`.
  - Tag: `scratch`.

- **`rho_flux_ann_params.nc`** (Nov 16, 12 kB) — produced by
  `generate_test_ann_params.ipynb`.
- **`logger_6Nov.nc`** (Nov 6, 42 kB) — produced by `train_ann_rho_test`.
- **`ann_instance_6Nov.nc`** (Nov 6, 28 kB) — ANN instance file; no notebook
  in this folder writes it explicitly, likely produced by the trainer in
  `src/training-on-CM2.6/`.

---

## Questions for Dhruv

1. **`PE_impacts.ipynb` vs `relative_mags_forcing.ipynb`** — these look like
   iterations on the same "magnitude of eddy forcing terms" figure. Which is
   the one you'd ship? (I've tagged `PE_impacts` as `unclear` and
   `relative_mags_forcing` as `canonical`.)
2. **`offline_eval.ipynb`** — this loads `ann_instance_20Dec.nc`
   (FGR3/EXP0). Is that the ANN used in the OM4/channel/NW2 deployment runs,
   i.e. the "final" model for the paper?
3. **`11-Dhruv-Nov-runs.ipynb`** — I tagged this `scratch` under the
   assumption `check_OM4.ipynb` (Apr 20) supersedes it. Confirm?
4. **`Figure-S3-and-Tables-S2-S3_DB.ipynb`** — does this regenerate figures
   for Part 1 (already published) or Part 2? Savefig paths point to Pavel's
   tree — does that notebook still need to run for Part 2?
5. **`NW2_analysis_R2/R3/R4`** — all three tagged canonical, but Part 2
   probably foregrounds one resolution. Which?
6. **`DG_compare_ANNC_20km.ipynb`** — is Double-Gyre in scope for Part 2, or
   is this a stale side-test?
7. **`train_ann_rho_test.ipynb`** — is training done from a `src/` module
   now, making this notebook archival? Or is it still the live trainer?

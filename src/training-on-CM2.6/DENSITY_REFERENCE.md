# Density reference for the ρ-flux ANN: σ₀ (offline) vs neutral (online/MOM6)

**Status: RESOLVED + DECISION TAKEN (2026-06-23). Matched control confirms working fully in neutral
costs ~zero offline skill in the flux band. → DECISION: the project adopts the NEUTRAL density
framework as canonical (exact train↔online consistency with MOM6's `calc_isoneutral_slopes`).**
Last updated 2026-06-23.

**Canonical model:** `EXP_neutral_all4` — neutral, all four factors [4,9,12,15] in-distribution,
canonical [32,32] config. **VALID as of 2026-06-24** (job 11630952). Flux R²F_away: f4 0.925, f9 0.839,
f12 0.779, **f15 0.719** (now in-distribution, vs cross-res 0.711). Matches EXP_neutral [4,9,12]
(0.929/0.840/0.776) → confirmed. Supersedes the σ₀ EXP0 as the production ρ-flux model.
(`EXP_neutral` [4,9,12] and `EXP_sigma0` [4,9,12] remain as the matched-comparison artifacts.)
**Canonical full skill profile** (flux / forcing / APE, depth-mean coast-excluded; forcing/APE via
`eval_force_ape.py` since the auto skill-test skipped the block on a transient `cm26.py` state during
concurrent edits):

| factor (Δ) | flux R²F_away | forcing R²F | APE R²F |
|---|---|---|---|
| 4 (0.4°) | 0.925 | 0.621* | 0.850* |
| 9 (0.9°) | 0.839 | 0.512 | 0.735 |
| 12 (1.2°) | 0.779 | 0.455 | 0.685 |
| 15 (1.5°) | 0.719 | 0.401 | 0.634 |

(*f4 forcing/APE NT=8-subsampled.) Matches EXP_neutral [4,9,12] (f9 force 0.512 vs 0.526, ape 0.735
vs 0.736) and adds f15 in-distribution. Flux > APE > forcing hierarchy holds at every resolution.
> ⚠️ **Being re-trained (2026-06-24) — two gotchas hit, both fixed:**
> 1. **Corrupt data → NaN training.** First all-4 run (job 11597160) diverged to NaN: adding factor-15
>    pulled in a corrupt all-NaN snapshot `subfilter-neutral/factor-15/train-67.nc` (quota-crash artifact;
>    the generator's skip-check validated only the `yq` coord, not data). Fixed: deleted + regenerated
>    train-67 (verified finite), and **hardened `generate_density_flux.py`** to require a non-all-NaN `Fx`
>    before skipping. Training then converged cleanly (MSE ~0.17–0.49, no NaN).
> 2. **`export_ANN` interactive prompt → batch crash.** Retrain (job 11624560) trained fine but the job
>    FAILED at `export_ANN` (`ann_tools.py:173–176`): when the model file already exists it prints
>    "Rewrite ... ?" and calls `input()` → `EOFError` in a batch job (no stdin), *before* writing. So the
>    good model was never saved. **Fix when retraining to an existing PATH_SAVE: delete the old
>    `model/ann_instance.nc` (+ skill files) first** so `export_ANN` writes directly with no prompt.
>    (Run 1 only "succeeded" because the path was new.) Cleared stale files; retrain v2 = job 11630952.
> EXP_neutral [4,9,12] was unaffected (never read factor-15). The flux/forcing/APE *comparison* results
> above are unaffected (they use EXP_neutral [4,9,12] / EXP0, not the all-4 model).

**Follow-ups to fully "switch" (not yet done):**
- Export `EXP_neutral_all4` for MOM6 via `export_ANN` (the .nc MOM6 reads) → online model becomes neutral.
- Repoint `offline_eval`/paper §3.3 headline offline numbers to the neutral model.
- PLAN.md M1 entry (other session owns the file): record decision = adopt neutral.

This note records *why* there are two coarse-grained training datasets that differ only in the
density reference, how the "neutral" one is built to match what MOM6 actually does, and what the
offline-skill comparison shows. Written so the work survives a lost session — a fresh reader (or
Claude) should be able to pick this up cold.

## TL;DR

- The ρ-flux ANN was originally trained on gradients/fluxes of **surface-referenced potential
  density σ₀** (`gsw.sigma0`, reference pressure p=0). But **MOM6 online builds the ANN's
  density-gradient input from the EOS derivatives at the *local* interface pressure** (a
  locally-referenced / "neutral" gradient, via `calc_isoneutral_slopes`). So train and online used
  a *different kind* of density gradient — they agree near the surface and diverge with depth
  (thermobaricity).
- Decision: **do it the way MOM6 does it.** Build a second, fully-consistent dataset in the neutral
  framework and retrain, so train ↔ online match exactly.
- Two datasets, identical pipeline, differing **only in reference pressure**:
  - **(1) σ₀**  → `subfilter/`            (potential density, p=0)
  - **(2) neutral** → `subfilter-neutral/` (locally-referenced, p=local — matches MOM6)
- **Result: working fully in neutral costs ~zero offline FLUX skill where it matters** — in the
  flux-active upper ocean (≤500 m holds 99.4 % of |F|²) neutral and σ₀ agree to within ±0.01 R²F at
  every resolution. So we can adopt the neutral pipeline for exact train↔online consistency at no
  flux-skill cost.
- **Stronger result on the APPLIED quantities (2026-06-24):** for the **forcing** `G=−u*·∇ρ` and
  **APE-sink** `a=Υ·∇_hρ` (what the scheme actually feeds back / the energetic sink), **neutral is
  meaningfully BETTER than σ₀ — +0.05–0.07 R² and +0.03–0.04 corr at f9/f12** (corr improves too →
  real, not a variance effect). Physical reason: forcing/APE go through `Υ=F_h/|∇₃ρ|` and so depend
  on the **vertical** gradient `∂_zρ`, where σ₀'s surface-referencing is wrong but neutral
  (`∂_zρ=−(ρ₀/g)N²`) is right. So neutral isn't just "free on the flux" — it improves what MOM6 applies.

## Background: the two reference frameworks

The subfilter ρ-flux target is `Fx = ρ̄·ū − (ρu)‾` (and likewise `Fy`); the ANN's density-gradient
inputs are `rhox, rhoy`; the dimensional output scaling uses `rho_norm = |∇ₕρ|`. Every one of these
depends on *which* density ρ you use:

- **σ₀ (potential density, ref p=0):** `ρ = gsw.rho(S,T, p=0)`. Surface-referenced; what
  `state.rho()` returns and what the original pipeline used. Gradient taken at constant model level.
- **neutral (locally-referenced, ref p=local):** `ρ = gsw.rho(S,T, p=local)`. The density whose
  *lateral gradient* defines the local neutral tangent plane. This is what GM/Redi and MOM6's
  isoneutral machinery use.

They are nearly identical near the surface (p→0) and **diverge with depth via thermobaricity**
(the pressure-dependence of the EOS). Measured on factor-9 CM2.6 test data, the neutral vs σ₀
gradient `rhox` has corr = 1.0000 through ~250 m, falling to ~0.6 and rms ratio ~4× by 4–5 km; the
flux target `Fx` shows the same pattern (corr 0.61, rms 6× at 5 km).

Terminology: what we call "neutral" here is **locally-referenced potential density** (EOS
derivatives at the local pressure) — the operational neutral gradient — *not* McDougall's neutral
density γⁿ. State it precisely in the paper.

## What MOM6 actually computes (verified in code)

Online code lives on branch **`rho_flux_ANN_gfdl_ready`** of Dhruv's MOM6 fork
(`/home/db194/MOM6-examples/src/MOM6/`). The ρ-flux driver feeds the ANN gradients from
`calc_isoneutral_slopes`:

- `MOM_meso_sfn_ANN.F90:196` → `call calc_isoneutral_slopes(... drdx_u=drdx_u, drdy_v=drdy_v ...)`
- Inside `MOM_isopycnal_slopes.F90`:
  ```fortran
  pres_u(I) = 0.5*(pres(i,j,K) + pres(i+1,j,K))                       ! local interface pressure  (:312)
  call calculate_density_derivs(T_u, S_u, pres_u, drho_dT_u, drho_dS_u, ...)  ! α,β AT local p     (:336)
  drdiB = drho_dT_u(I)*(T(i+1,j,k)-T(i,j,k)) + drho_dS_u(I)*(S(i+1,j,k)-S(i,j,k))  !                (:366)
  ```
  i.e. the lateral gradient is
  **∂ρ/∂x = (∂ρ/∂T)|ₚ · ∂T/∂x + (∂ρ/∂S)|ₚ · ∂S/∂x, with the EOS derivatives evaluated at the local
  interface pressure `pres_u`** (pressure held fixed across the horizontal difference). That is the
  locally-referenced (neutral) gradient — not σ₀.

## How the neutral dataset is built, and why it matches MOM6

`scripts/generate_density_flux.py` with `DENSITY=neutral`:

```python
p_ref = 0.0 if DENSITY=='sigma0' else ds.param.zl   # local pressure (zl ~ dbar) for neutral
rho   = gsw.rho(ts.salt, ts.temp, p_ref)            # overwrite rho with the chosen reference
ds.data['rho'] = rho
coarse = ds.compute_subfilter_forcing(...)          # UNCHANGED pipeline -> Fx/Fy, rhox/rhoy, N² all consistent
```

Because `p_ref = zl` is **horizontally uniform per level**, the horizontal gradient of
`gsw.rho(S,T,p=zl)` is identically `(∂ρ/∂T)|ₚ·∇T + (∂ρ/∂S)|ₚ·∇S` — **the same formula MOM6 uses**.
And it is not just the gradient: overwriting ρ and rerunning the unchanged flux pipeline makes the
flux *target*, `rho_norm`, and N² all consistently neutral. (N² is additionally recomputed the
locally-referenced way via gsw α/β at local pressure, because the pipeline's vertical N² diffs
in-situ ρ across levels of varying pressure → compression-contaminated; see the N²-input study.)

Two implementation differences from MOM6, both negligible or controlled:
1. **Pressure value:** MOM6 uses true hydrostatic pressure (∫ρg dz); we use p ≈ depth-in-dbar
   (`zl`). Standard approximation, ≲1 % off; α/β depend only weakly on it.
2. **EOS:** we use gsw (TEOS-10); MOM6 uses its configured EOS. The σ₀ dataset *also* uses gsw, so
   this is held constant in the σ₀-vs-neutral comparison (it is a separate, pre-existing train↔online
   EOS question, not the reference-pressure question studied here).

## The two datasets (on disk)

Root: `/scratch/$USER/CM26_datasets/ocean3d/`. Both built through the identical
`generate_density_flux.py` pipeline; trainable via `read_datasets(subfilter='subfilter[-neutral]')`.

| dataset | density | path | status (2026-06-23) |
|---|---|---|---|
| (1) σ₀ | potential, p=0 | `subfilter/FGR3/factor-{4,9,12,15}` | complete, all 4 factors (~804 GB) — also EXP0's training data |
| (2) neutral | local p (matches `calc_isoneutral_slopes`) | `subfilter-neutral/FGR3/factor-{4,9,12,15}` | 4/9/12 complete; **f15 train 68/96** (`genneu_tr15` queued) |

> NB the σ₀ data already existed as the production `subfilter/` set — there is **no need to
> regenerate it** as `subfilter-sigma0`. (A `subfilter-sigma0` regen was briefly queued for
> pipeline-identity rigor, then cancelled as redundant; the matched σ₀ control is instead obtained
> by training on the existing `subfilter/` set with the same factor set as the neutral model.)

**Variable completeness:** `generate_density_flux.py` runs the *unchanged* `compute_subfilter_forcing`
(with `add_rho_fluxes=True`), so each neutral file is a **strict superset** of a production file —
all 42 production variables **plus** stored `temp`/`salt` (44 total). That includes the full momentum
suite: stress tensor `Txx/Txy/Tyy`, forcing `SGSx/SGSy(_h)`, energetics `SGS_KE/SGS_diss(_deviatoric)`,
and every velocity-gradient diagnostic. The momentum terms are **density-independent** (built from
`u,v`), so they are *identical* between `subfilter/` and `subfilter-neutral/`; the two sets differ in
**only** the density-derived fields (`rho, rhox/rhoy, rho_grad_mag, Fx/Fy, N_buoyancy`, and quantities
built from them: `deformation_radius, eady_time, dudz_geo*`). The neutral set is thus fully usable for
momentum work too.

## Results

### A. Clean cross-eval — the actual online scenario (no retrain)
`scripts/eval_cross.py`: take the σ₀-trained model (the one deployed) and feed it the **neutral**
gradient — exactly what happens online. Factor-9 test, band-mean R²F:

| depth band | σ₀ input | neutral input | drop |
|---|---|---|---|
| 0–500 m (99.4 % of \|F\|²) | 0.801 | 0.785 | **−0.015** |
| 500–1500 m | 0.828 | 0.791 | −0.037 |
| 1500–5500 m (<0.6 % of \|F\|²) | 0.561 | −6.48 | catastrophic, but negligible flux + caught by online clamps |

→ The train↔online mismatch costs only **−0.015 R²F in the flux band**. Negligible where it matters.
Corroborates the Tier-1 WOA climatology diagnostic (gradient rotation 0.12°, magnitude ratio 1.008).

### B. Fully-neutral retrain vs σ₀ — the right comparison
`EXP_neutral` (canonical [32,32], factors [4,9,12], whole pipeline neutral; job 11565805) scored on
the neutral test target, vs σ₀ EXP0. Depth-mean R²F_away and upper-ocean (≤500 m):

| Δ / factor | neutral (depth-mean) | σ₀ (depth-mean) | **neutral ≤500 m** | **σ₀ ≤500 m** |
|---|---|---|---|---|
| 0.4° / 4  | 0.929 | 0.848 | **0.946** | **0.942** |
| 0.9° / 9  | 0.840 | 0.781 | **0.870** | **0.872** |
| 1.2° / 12 | 0.776 | 0.725 | **0.819** | **0.828** |
| 1.5° / 15* | 0.711 | 0.665 | **0.766** | **0.780** |

*neutral cross-resolution at f15 (trained on 4,9,12 only). σ₀ = EXP0 (`skill-test-rho[-fixed]`).

- **In the flux band (≤500 m): neutral ≈ σ₀ to within ±0.01 at every resolution.** Working fully in
  neutral costs no offline skill where the flux lives.
- The depth-mean is ~0.05–0.08 *higher* for neutral, but **do not overclaim "neutral is better"** —
  each model is scored on its own target and R² is target-variance-relative; the neutral flux simply
  carries more coherent variance in the deep ocean (where the targets diverge and flux is
  negligible). The decision-relevant number is the upper-ocean tie.

### B′. Matched σ₀ control — EXP_neutral vs EXP_sigma0 (the airtight comparison)
`EXP_sigma0`: same σ₀ data (`subfilter/`) and canonical config as EXP0, but trained on factors
**[4,9,12]** to match `EXP_neutral` (job 11573247, completed). So the two models differ in
essentially just the reference pressure. Upper-ocean (≤500 m) and depth-mean R²F_away:

| Δ / factor | neutral ≤200 m | σ₀ ≤200 m | **neutral ≤500 m** | **σ₀ ≤500 m** | Δ(≤500 m) | neutral dm | σ₀ dm |
|---|---|---|---|---|---|---|---|
| 0.4° / 4  | 0.941 | 0.941 | **0.946** | **0.946** | −0.001 | 0.929 | 0.854 |
| 0.9° / 9  | 0.858 | 0.858 | **0.870** | **0.870** | −0.000 | 0.840 | 0.783 |
| 1.2° / 12 | 0.812 | 0.812 | **0.819** | **0.820** | −0.000 | 0.776 | 0.721 |
| 1.5° / 15*| 0.761 | 0.759 | **0.766** | **0.763** | +0.003 | 0.711 | 0.655 |

*both cross-resolution at f15 (trained on [4,9,12]).

**Flux-band skill is identical to ±0.003 at every resolution** — the upper-ocean values match to
three decimals because neutral ≈ σ₀ there (corr 1.0). `EXP_sigma0`'s depth-mean (0.854/0.783/0.721/
0.655) ≈ EXP0's (0.848/0.781/0.725/0.665) → the factor-set difference *and* the production-vs-
`generate_density_flux` pipeline difference are both immaterial (σ₀ gradient matches across pipelines
at corr 0.9999). **This is the airtight confirmation: σ₀ and neutral are interchangeable in the
flux-active ocean.**

### B″. Depth structure of the difference (where, and is it real?)
Depth-resolved EXP_neutral vs EXP_sigma0 (factor-9), reporting **both** R²F_away and corr_F (to
separate real predictability from target-variance), with the deep flux weight:

| band | R²F neu/σ₀ | corr neu/σ₀ | var_neu/var_σ₀ | % of \|F\|² below band-top |
|---|---|---|---|---|
| 0–500 m | 0.870 / 0.870 | 0.896 / 0.897 | 1.0 | — |
| 500–1500 m | 0.880 / 0.883 | 0.916 / 0.918 | 1.3 | 0.19 % (σ₀) / 0.28 % (neu) below 1000 m |
| 1500–3000 m | 0.829 / 0.827 | 0.877 / 0.874 | 2.2 | 0.01 % below 2000 m |
| 3000–5500 m | 0.761 / 0.524 | 0.789 / 0.676 | 10.4 | ~0 % below 3000 m |

- **Down to ~3000 m: interchangeable** — both R² *and* corr tied to ≤0.005, even as the neutral flux
  variance grows to 2–5×. The density variable does not affect skill where the flux lives.
- **Below ~3000 m: neutral genuinely more skillful** (5185 m: R²F 0.79 vs 0.29, corr 0.81 vs 0.51).
  **Correlation improves, not just R² → this is real predictability, NOT a variance artifact**
  (an earlier note wrongly dismissed it as variance-only — corrected here). Physically: σ₀'s
  surface reference makes its abyssal gradient thermobarically ill-posed/noisy; locally-referenced
  (neutral) density is the meaningful variable at depth. This abyssal gain is the *entire* source of
  the higher neutral depth-mean (~13 deep levels at +0.25; everything above 3000 m ties).
- **Dynamically negligible:** the abyss carries <0.3 % of |F|² below 1000 m, ~0 % below 2000 m. So
  the deep gain is a clean physical point in favour of neutral, not a consequential skill change.

**Holds at ALL coarsening factors (checked 2026-06-23).** The three-part pattern is robust across
f4/9/12/15. 0–500 m R²F (neu/σ₀): 0.946/0.946, 0.870/0.870, 0.819/0.820, 0.766/0.763 — tied. Abyssal
3000–5500 m R²F: 0.883/0.569, 0.761/0.524, 0.684/0.464, 0.604/0.390; **corr**: 0.871/0.715, 0.789/0.676,
0.742/0.634, 0.697/0.580 — **Δcorr = +0.16/+0.11/+0.11/+0.12, positive and ≈scale-invariant → the
abyssal neutral advantage is real predictability (corr, not just R²) at every scale.** R² gap largest at
finest grid (f4 +0.31), eases with coarsening, never flips. Deep flux weight <0.1 % below 1500 m at every
factor. f15 is cross-resolution for *both* models (neither trained on it) and the pattern still holds.
(Script: `/scratch/$USER/deep_compare_allfactors.py`.)

### C. Forcing & APE-sink skill — the quantities the scheme actually applies (2026-06-24)
Beyond the flux and its horizontal divergence, `SGS_skill_rho` (cm26.py:786–825) now also scores the
quantities the scheme feeds back, built from the predicted flux exactly as `MOM_meso_sfn_ANN.F90` does
(audited: clamps `FLUX_CLAMP=1e2`, `UPS_CLAMP=15`, `GRAD_FLOOR=1e-10` and per-component
`|∇₃ρ|=√(ρ_x²+ρ_z²)` all match the Fortran defaults; `∂_zρ=−(ρ₀/g)N²` from the locally-referenced N²):
- **Υ** eddy-induced transport `= F_h/|∇₃ρ|` (Ferrari 2010);
- **APE-sink** `a = Υ·∇_hρ` → `R2F_ape`, `corr_F_ape_away`;
- **Forcing** `G = −u*·∇ρ`, `u*=(∂_zΥ, −div_hΥ)` → `R2F_force`, `corr_F_force_away`.

Why these differ from the flux: both go through Υ and so depend on the **vertical** gradient `∂_zρ`
(and `G` on `∂_zΥ`). σ₀'s `∂_zσ₀` is surface-referenced (thermobarically wrong at depth); neutral's
`∂_zρ` is the true local stratification. So once a metric touches the vertical structure, neutral wins.

Matched `EXP_neutral` vs `EXP_sigma0` ([4,9,12], same config), depth-mean coast-excluded
(`skill-test-forceape/`; produced by `scripts/eval_force_ape.py`):

| metric | f4 (NT=8) neu/σ₀ (Δ) | f9 neu/σ₀ (Δ) | f12 neu/σ₀ (Δ) |
|---|---|---|---|
| R²F_force | 0.647 / 0.553 (**+0.093**) | 0.526 / 0.462 (**+0.064**) | 0.453 / 0.400 (**+0.054**) |
| corr_F_force_away | 0.833 / 0.772 (+0.061) | 0.756 / 0.716 (+0.040) | 0.699 / 0.668 (+0.032) |
| R²F_ape | 0.859 / 0.755 (**+0.104**) | 0.736 / 0.670 (**+0.066**) | 0.677 / 0.614 (**+0.063**) |
| corr_F_ape_away | 0.928 / 0.871 (+0.057) | 0.860 / 0.824 (+0.036) | 0.823 / 0.790 (+0.033) |

(EXP0 σ₀ all-4 corroborates EXP_sigma0: f4 0.525/0.742, f9 0.446/0.666, f12 0.392/0.616, f15 0.334/0.565.)
**The neutral advantage grows toward finer resolution** (force Δ +0.093→+0.054, ape +0.104→+0.063 as
f4→f12) — at finer Δ the vertical structure entering Υ is better resolved, so the σ₀-vs-neutral
vertical-gradient difference shows up more. f4 is NT=8-subsampled (heavy); both models use the same
subsampling so the *gap* is apples-to-apples (absolute f4 values may shift slightly with full sampling).

**Hierarchy (f9, neutral):** flux R²≈0.84 > APE 0.74 > forcing 0.53 — the network reproduces the flux
best, the energetic sink second, the applied tendency least (G is derivative-heavy: ∂_zΥ, div_hΥ amplify
small scales, like the flux divergence). **Takeaway: neutral is tied with σ₀ on the flux but +0.05–0.07
R² better on the forcing/APE — it improves what MOM6 actually feeds back.**

**Caveat (for a bulletproof paper number):** the σ₀ side uses the production `subfilter` set's N²
(`g∂_zσ₀/ρ₀`), while neutral N² was recomputed via gsw α/β — so a *small* part of the gap could be
N²-method rather than pure reference-pressure. Direction + cross-factor consistency + corr improvement
argue it's genuinely physics; the airtight version needs the (cancelled) `subfilter-sigma0` set with a
gsw σ₀ N². (All factors 4/9/12 done; f4 time-subsampled NT=8 — gap apples-to-apples.)

### C. What NOT to do — the confounded append-pilot (cautionary)
An earlier "pilot" (`append_rho_kinds.py` + `--rho_grad_source neutral`) swapped *only the gradient
input* to neutral while keeping the **σ₀ flux target**. This produced an apparent ~0.30 R²F "drop"
(neutral 0.38 vs σ₀ 0.72 depth-mean, factor-9) that is a **training artifact, not a real cost**:
- The neutral model fits worse at *every* depth including the surface (5 m train-MSE 0.54 vs 0.25),
  yet the surface neutral & σ₀ inputs are corr=1.0000 identical (same seed/init/target) → impossible
  unless the shared MLP is compromised.
- Mechanism: overwriting `rhox/rhoy` also changed `rho_norm` to the neutral gradient (up to 4× larger
  at depth) **while the flux target stayed σ₀** → deep slices internally inconsistent → the single
  shared MLP trades off across all depths. The fully-neutral pipeline (B) avoids this because the
  target and `rho_norm` inflate together at depth, keeping the dimensionless NN target well-behaved.

## Decision / implication

We have empirical license for the **cleanest** resolution of the train↔online gradient-kind issue:
**move the whole pipeline to neutral density** (matching `calc_isoneutral_slopes` exactly) at no
offline-skill cost — rather than patching a σ₀-gradient path into the Fortran. Pending the EXP_sigma0
matched control as final confirmation.

## Artifacts

- **Datasets:** `subfilter/` (σ₀), `subfilter-neutral/` (neutral) under `/scratch/$USER/CM26_datasets/ocean3d/`.
- **Generator:** `scripts/generate_density_flux.py` (`DENSITY=sigma0|neutral`), slurm `scripts/slurm_generate_density.sh`.
- **Trainer:** `scripts/train_script_rho_fluxes.py` (`--subfilter`); slurm wrapper `scripts/slurm_train_subfilter.sh`
  (forwards `--subfilter`, unlike `slurm_train_ann.sh`).
- **Models:** `/scratch/$USER/mom6/CM26_ML_models/FGR3/` → `EXP0` (σ₀ canonical), `EXP_neutral`
  (fully neutral), `EXP_sigma0` (matched σ₀ control, [4,9,12]).
- **Eval:** `scripts/eval_cross.py` (σ₀ model + neutral input — the online scenario),
  `scripts/eval_rho_kinds.py` (matched per-kind eval). Read/compare helpers in `/scratch/$USER/*.py`.
- **MOM6 reference:** `MOM_meso_sfn_ANN.F90`, `MOM_isopycnal_slopes.F90` (`calc_isoneutral_slopes`) on
  branch `rho_flux_ANN_gfdl_ready`.

## Open / next

- Fill in EXP_sigma0 matched-control numbers (Section B).
- Optional: finish `subfilter-neutral` factor-15 train (68→96, `genneu_tr15`) → rerun `EXP_neutral`
  on all four factors so f15 is in-distribution rather than cross-resolution.
- Decide whether the production/online model and the paper's headline offline numbers switch to the
  neutral framework (recommended) or stay σ₀ with this note as the justification that it doesn't
  matter in the flux band.

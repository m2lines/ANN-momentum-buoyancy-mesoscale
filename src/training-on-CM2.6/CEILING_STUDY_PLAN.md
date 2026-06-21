# Predictability-ceiling study — plan (Part 2, CM2.6 3-D buoyancy fluxes)

The offline skill of the canonical ρ-flux ANN plateaus at **R²F ≈ 0.73** (depth-mean, test
split). Two independent saturation curves — capacity ([16,16]→[64,64] = 0.66→0.73) and footprint
(1×1→5×5 = 0.55→0.73) — both wall out near 0.73, so we are **not model-limited** for the current
5-feature, 3×3 input. The open question: is 0.73 the **predictability ceiling** of the resolved
coarse state, or are we **input-limited** (richer inputs would recover more)?

Goal: split the 0.27 of unexplained variance into three buckets —
**model shortfall · recoverable (missing-input) information · irreducible stochastic floor** —
and decide whether the path forward is *bigger/richer inputs* or *a stochastic closure*.

## Core idea: bracket the ceiling from below and above

Let R²_max = best R² any deterministic function of the **resolved coarse state** can achieve.
- **From below** — grow inputs + model until skill saturates. The asymptote is a lower bound on
  R²_max. Taken to its maximum (the full 3-D cube, below) it is the tightest such bound.
- **From above** — a model-free nearest-neighbour estimate of the irreducible noise variance on
  the same inputs (no trained model needed).
Where the two meet **is** R²_max. The gap (R²_max → 1.0) is the irreducible floor; the gap
(0.73 → R²_max) is what richer inputs can buy.

### Key conceptual point (why the cube is decisive, and why it can't reach 1.0)
A 3-D cube of coarse velocity + density + topography around the target column is a strict
**superset of the resolved information** — our gradients/strain/vorticity are linear functionals
of it, plus it carries vertical structure, wider footprint, raw values, stratification, boundary
proximity. But it is *all the information at the **filtered** scale* — **not** all the information
that physically set the flux. Filtering removed the sub-filter eddies that produced the subfilter
flux; two byte-identical coarse cubes can still have different true fluxes (the one-to-many closure
problem). So even the maximal-input model **cannot reach R²=1** — and that is exactly what makes it
useful: where it saturates is R²_max, and the remainder to 1.0 is the irreducible floor, measured
not assumed.

## Phase 0 — Freeze the evaluation substrate (½ day, no new science)
- **Depth-band stratification.** The 0.73 depth-mean is dragged down by low-variance abyssal
  levels; the ceiling is depth-dependent. Report R²_max in the band where the flux is
  non-negligible *separately* from the depth-mean. The headline question only makes sense per band.
- **Dump the held-out table.** On the test split (model years 199–200), write
  `(input-vector, true-flux Fx/Fy, model-prediction)` per (factor, depth) in the **normalized**
  input space the ANN sees. One artifact, feeds every later phase with no recompute. Extend it with
  the cube inputs (Phase A) and the unused fields (Phase B) as those come online.

## Phase A — From below: push the deterministic frontier to its maximum
The centrepiece is the **full-cube model**; the cheap MLP corners are warm-ups / sanity anchors.

**A1 — cheap MLP frontier (reuses sweep infra, a few GPU-hours).** 7×7 stencil and the untested
5×5 × [64,64] corner. Extends the existing 0.55→0.70→0.73 curve; tells us whether footprint × width
together still meet the same wall.

**A2 — full-cube model (centrepiece; new code).** Feed a 3-D neighbourhood of coarse velocity +
density + topography around the column and train the strongest tractable model. This is the maximal
**resolved** input — its saturation is the tightest lower bound on R²_max.
- **Architecture = weight-sharing (3-D CNN / conv over the cube), NOT a dense MLP.** A cube is
  thousands of inputs; a dense net on monthly snapshots would be underdetermined. Conv weight-
  sharing respects spatial structure and cuts parameters by orders of magnitude, making the fit
  tractable on our data.
- **This is an offline diagnostic, not a proposed online closure** — a cube is far too expensive
  inside MOM6. Its job is to measure *how much* is recoverable and *which physics* carries it; the
  actionable output is a few targeted inputs (Phase C), not a shipped cube.

### Acceptance criteria for A2 (these decide whether the number is trustworthy)
The trap is **data-limitation**: a giant-input model can plateau *low* because it can't be fit, not
because the information is absent — which would *under*estimate R²_max and fool us into "we're at
the ceiling." A2's saturation is only valid if:
1. **train ≈ test** (small generalization gap). If train ≫ test, it's data-limited → the number is
   meaningless; regularize / shrink the receptive field / report it as a lower bound only.
2. The model-free estimate **B1 run on the cube inputs agrees** with where A2 saturates (brackets
   the ceiling from both sides → data-limitation worry closed).
3. Monotonicity: A2 ≥ the A1 frontier ≥ the 3×3 baseline on the same split (a richer input can't
   *lose* if optimization is sound — a violation flags a bug to chase, not a ceiling).

## Phase B — From above + discriminate (cheap, offline, model-free)
**B1 — nearest-neighbour noise estimate (Gamma/delta test).** For near-duplicate input vectors
across space and time, the spread of their *true* fluxes is the irreducible conditional variance
(identical coarse state, different flux = the one-to-many problem made measurable). KDTree on the
Phase-0 table → Var(noise) → R²_ceiling, model-free. **Run per input set** (1×1/3×3/5×5/7×7 stencils
*and* the cube): the ceiling-vs-footprint curve quantifies recoverable non-locality decoupled from
model capacity. Overlay on model-R²-vs-footprint — gap at each footprint = model shortfall, rise
with footprint = recoverable signal.

**B2 — residual-structure analysis.** Does the trained model's error correlate with **resolved
fields we don't feed it** — N²/stratification, larger-scale strain, local KE, divergence, vertical
shear? Structured residual → input-limited, and the correlating field *names* the missing input.
White, field-uncorrelated residual → we're at the stochastic floor.

## Phase C — Close the loop (only if A2/B2 say recoverable)
Add the top residual-correlated field(s) named by B2 (e.g. N², a wider stencil, vertical structure)
as cheap explicit inputs to the *local* ANN, retrain, check R² moves toward R²_max. Converts "the
cube wins / the residual correlates with N²" into a confirmed, online-affordable gain — and into a
paper paragraph: *which* few inputs buy back *how much* of the cube's advantage.

## Decision logic
| A-frontier (esp. A2) | B1 ceiling | B2 residual | Verdict |
|---|---|---|---|
| ≈ 0.73 | ≈ 0.73 | white | **At the ceiling.** Residual is irreducible → path forward is *stochastic*, not bigger inputs. |
| ≫ 0.73 (train≈test) | ≫ 0.73 | structured | **Input-limited.** Phase C buys it back with named inputs (non-locality / stratification). |
| ≈ 0.73 but A2 train ≫ test | ≫ 0.73 | — | **Inconclusive — A2 is data-limited.** Regularize / shrink; trust B1 as the ceiling for now. |
| < B1 | ≫ 0.73 | — | Model-limited (inconsistent with the plateau → flags a bug). |

Note: R²_max measured this way is *in-distribution*. A model that must also **generalize** (non-dim,
rotation) sits a bit below it — consistent with the Phase-2 rotation finding (costs in-distribution
skill, buys cross-config generalization).

## Cost & ordering
1. **Phase 0 + B (offline, CPU-minutes on the dumped table).** Front-loads the answer model-free
   before any GPU spend. B1's footprint curve + B2's residuals may settle it on their own.
2. **A1 (a few GPU-hours, existing sweep infra).** Cheap confirmation of the MLP wall.
3. **A2 (new CNN code, the main build).** Only worth it if B leaves the recoverable-vs-irreducible
   split ambiguous — which is the likely case worth resolving rigorously.
4. **C** gated on A2/B2 naming a field.

## Phase B results (2026-06-20) — DONE (with a methodological caveat)
Offline, model-free front-load on the baseline `st3_h32-32_s0`, test split, 4 factors, 40k
points per (factor, depth), 8 snapshots. Scripts: `scripts/{dump_ceiling_table,
ceiling_nn_estimate,ceiling_residual_structure}.py`; table on scratch (`ceiling/table_all.npz`).
The dump reproduces the canonical dimensional skill exactly (R2F 0.80/0.73/0.67/0.61) — faithful.

**B1 (nearest-neighbour / Gamma ceiling) is UNRELIABLE for this field — a real negative result.**
The estimator is squeezed between two biases with no clean middle:
- include input-space neighbours → they are often spatial/temporal neighbours whose flux is
  correlated for reasons unrelated to the inputs (adjacent cells even share stencil values) →
  flux-gap too small → ceiling biased HIGH (the spurious +0.12 headroom we first saw);
- exclude them (cross-time AND cross-location, as we ended up doing) → the nearest admissible
  neighbour is forced too far in input space → flux-gap too large → noise overestimated →
  ceiling biased LOW.
The low-bias dominates at fine resolution: there the model R2F (0.89 upper-ocean at 0.4°)
*exceeds* even the extrapolated ceiling (0.82) — impossible for a true ceiling, so it proves
the estimate is biased low. Heavy tails + 45-dim sparsity + strong field autocorrelation make
the model-free ceiling untrustworthy as an absolute number. **Do not report a model-free ceiling.**

Upper-ocean (<1500 m) depth-mean: model 0.891/0.819/0.766/0.712 (factors 4/9/12/15);
cross-time Gamma bracket [lb, extrap] = [0.67,0.82]/[0.65,0.81]/[0.65,0.81]/[0.64,0.80].
The bracket straddles the model only at coarse resolution; at fine resolution the model is above it.

**What IS robust:**
1. Model R2F is itself a hard lower bound on the ceiling: upper-ocean ≥ 0.89/0.82/0.77/0.71.
   Unexplained variance grows monotonically with coarsening (11% → 29%) — consistent with
   coarser filtering destroying more sub-filter information → a LARGER irreducible stochastic
   floor, not a model deficiency.
2. **B2 (scale-removed residual |T-P|^2):** only weakly structured against the discarded
   magnitudes / KE (eta^2(log) ≈ 0.09–0.29, consistent across all 4 factors); the signal is the
   mild "weak-gradient regions are relatively harder" effect (negative spearman, strongest in the
   abyss). No strong evidence that a cheap extra LOCAL input (KE) would close the gap.

**Conclusion / pivot:** the model-free route cannot pin the ceiling for this autocorrelated,
high-skill field. The trustworthy instrument is the SUPERVISED from-below approach (Phase A):
train progressively richer models on a train split and read where held-out skill saturates —
train/test immunizes it against the autocorrelation that sank B1. Capacity is already known to
plateau (~0.73 cross-res; not model-limited), so the decisive remaining axis is FOOTPRINT:
5×5 → 7×7 → cube/CNN. If held-out skill saturates near the current model, we are at the ceiling
(measured cleanly this time); if the cube jumps, we were footprint/non-locality-limited.
Recommended next: the cheap MLP footprint corners (7×7, 5×5×[64,64]) via the existing sweep
infra, before building the CNN.

## Phase A2 results (2026-06-20) — DONE: cube/CNN probe, negative result
The supervised from-below probe. A full-column 2-D CNN (`scripts/train_cube_cnn.py`,
`slurm_train_cube.sh`): the whole depth column of each field stacked as channels (vertical
coupling via 1x1 mixing) + a dilated 3x3 stack giving a **31x31** horizontal receptive field
(~12 deg at 0.4 deg up to ~47 deg at 1.5 deg -- gyre/basin scale). Strict superset of the
3x3 MLP: adds raw velocity, density, N_buoyancy, the discarded magnitudes, full column, basin
footprint. Output = dimensionless flux x s (s=|grad rho|*|grad u|*dx^2). Scored through the
SAME SGS_skill_rho as the MLP. Several bugs fixed en route (H200/torch-cu117 mismatch -> L40S;
FIELDS comma/--export collision -> colons; missing dx^2 in s -> frozen loss=BATCH; OOM ->
CPU-resident data + per-batch GPU + MAXTEST cap).

Upper-ocean (<1500 m) test R2F:

| res   | MLP   | cube (standardized) | cube (MLP non-dim) | cube non-dim *train* |
|-------|-------|---------------------|--------------------|----------------------|
| 0.4   | 0.891 | 0.691               | 0.660              | 0.668                |
| 0.9   | 0.819 | 0.659               | 0.616              | 0.684                |
| 1.2   | 0.766 | 0.624               | 0.575              | 0.683                |
| 1.5   | 0.712 | 0.594               | 0.544              | 0.667                |

Findings:
- **The cube underperforms the per-point MLP at every resolution**, and is negative all-depth
  (collapses in the abyss) where the MLP is positive (0.61-0.80).
- Single-factor cube overfit (train 0.75 / test 0.62 at f15); **pooling all 4 factors closed
  the gap** (train~test) but did not raise test skill.
- **Option B (porting the MLP's per-point Frobenius non-dimensionalization) did NOT help -- it
  was slightly worse.** Decisive: even with the MLP's own representation, the cube cannot FIT
  train past ~0.67 while the MLP generalizes to 0.89. So the gap is **not** the input
  representation. (B < A is itself the rotation lesson again: non-dim discards magnitude info
  that helps in-distribution.)
- Conclusion: the bottleneck is the **field-CNN learning paradigm**, a weaker approximator than
  the per-point MLP for this problem -- not footprint, data, or representation. The CNN-cube
  **cannot serve as a from-below ceiling probe** without serious ML engineering (per-depth heads,
  3-D conv, architecture search) -- a separate project. Route closed for this paper.

## Overall conclusion (Phase B + A2)
No clean *measured* ceiling (model-free B1 unreliable here; cube underperforms the baseline). But
the trustworthy evidence converges: capacity plateaus (~0.73; MLP not model-limited), the
scale-removed residual is only weakly structured against unused fields (B2), unexplained variance
grows monotonically with coarsening (irreducible sub-filter floor), and a basin-scale full-column
cube with the MLP's representation shows **no accessible headroom**. Read: **the MLP is near the
predictability ceiling of the local resolved state; the gap to perfect is dominated by irreducible
sub-filter stochasticity that grows with coarsening.** Decision: **keep the MLP for Part 2**; the
cube / richer-architecture investigation is **future work**.

## Outputs
- `DB_notebooks/evaluate_model_design/predictability_ceiling.ipynb` — the bracket figure
  (model-R² vs footprint, B1 ceiling vs footprint, A2 saturation, irreducible floor) + residual
  structure. Scratch-only + regenerate, mirroring the sensitivity study.
- A quantified split of the 0.27 gap → a §3.x / discussion result either way:
  *irreducible floor → "future work is stochastic"*, or *input-limited → "N²/non-locality is the
  next input, worth +X."*
- Decision on whether to extend the canonical input set (Phase C).

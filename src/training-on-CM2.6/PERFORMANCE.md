# Training & offline-skill performance work (rho-flux pipeline)

Record of the optimization pass on the rho-flux training pipeline (`train_rho_fluxes.py`,
`ANN_rho_inference`, `predict_ANN_rho`) after the Greene -> NYU-torch cluster migration.
Every change was validated against the pre-optimization code; the science is unchanged.

## TL;DR

| Pipeline stage | before | after | bit-exact? |
|---|---|---|---|
| **Training loop** (per iter) | ~46.5 s (lazy, validate every step) | ~0.70 s (preload + throttle + GPU) | weights identical for CPU step; full GPU run *equivalent* |
| **Training, end-to-end run** | ~7 h | ~42 min (~10x) | reproduces baseline skill |
| **Offline skill / test eval** | ~15-25 min (per-slice CPU) | ~10x on GPU (batched) | yes (max abs diff 0.0) |

The headline "~55x per training iteration" is the product of two clean,
same-hardware factors (below). End-to-end it is ~10x, because dataset read + the
test/skill block do not speed up proportionally (Amdahl).

## What was slow, and why (measured, not guessed)

Profiling one in-memory factor-4 training step (`cProfile`, 346k wet points):

- **~⅔ of each step was disk I/O.** `read_datasets(load=False)` keeps data lazy, so
  every step re-read 9 variables off disk via `tensor_from_xarray -> .to_numpy()`.
  factor-4's 442 GB train set can't sit in page cache, so reads stayed disk-bound.
- The remaining compute split **~42% ANN (fwd+bwd) / ~54% feature marshalling
  (torch `unfold`/`pow`/`concat`/`cast`) / ~3% host xarray.** The xgcm grid build,
  which we first suspected, was ~0 ms.

## The optimizations

1. **Preload to RAM** (`train_rho_fluxes.py`). Load once the 8 variables the step
   actually touches; every step is then RAM-fast. ~3x on factor-4, ~8x on small
   factors. **Bit-identical** to lazy.
2. **Throttle validation** (`--validate_every`, default 10). Validation is `no_grad`,
   so a full pass every step only doubled cost for a denser curve than needed.
   `--validate_every 1` restores the original every-step behaviour exactly.
   - (1)+(2) together: **46.5 -> 6.2 s/iter = 7.5x**, same `cs` hardware, bit-exact.
3. **GPU** (`--device cuda`). The whole per-step pipeline -- stencil/norm/concat
   feature build *and* ANN fwd/bwd -- runs on device (the marshalling is torch ops,
   so the GPU captures it too, not just the matmul). Same-node (l40s) **5.2 -> 0.70
   s/iter = 7.4x**. The training loop never touched CUDA before; the old `--nv` in
   the Greene slurm script was cargo-culted.
4. **Batch `predict_ANN_rho` over time** (the offline skill path). The per-slice
   predict was bound by Python/launch/sync overhead (1200 tiny forwards), so plain
   GPU gave only ~1.2x and CPU-batching gave nothing. Doing **one batched forward per
   depth** feeds the GPU -> **~10x**. Implemented as a *shared* batch-capable
   `ANN_rho_inference` (carries an optional leading time dim), so single-slice
   training and batched predict are one code path. **Bit-identical** (0.0).

## Bit-exactness vs equivalence

- preloaded training **==** lazy training: weights max abs diff **0.0**.
- batched predict **==** per-slice predict: **0.0**.
- single inference CPU **==** GPU: **0.0**.
- A *full* 500-iter CPU-vs-GPU run is **equivalent, not identical** (test R2F differs
  at the 3rd decimal, e.g. 0.726 vs 0.727). This is expected: float addition is
  non-associative and CPU/GPU run reductions in different orders, which compounds over
  ~50k weight updates. Both seed-0 runs reproduce the canonical model's skill
  (0.80 / 0.73 / 0.67 / 0.61) to 2 decimals.

## Running the *original* (pre-optimization) approach

The original outputs are always reproducible (bit-exact, above). The original *code
path* lives in git, not behind a runtime flag:

- last pre-optimization state: **commit `772c23b`** (lazy training + per-slice predict
  + validate-every-step; already torch-cluster-runnable).
- to run it: `git checkout 772c23b -- helpers/train_rho_fluxes.py helpers/cm26.py`
  (then `git checkout HEAD -- ...` to restore), or branch off `772c23b`.
- In the current code, CPU (`--device cpu`, default) and every-step validation
  (`--validate_every 1`) are recoverable directly; the RAM preload is unconditional
  and predict is batched (no per-slice toggle) -- both bit-exact equivalents.

## Cluster / container notes (NYU torch)

- Partitions: `cs` (CPU), and GPU: `l40s_public`, `h100`, `h200_public`,
  `rtx6000_lzanna` (Zanna group). `cs` is CPU-only.
- Container: `/scratch/$USER/Pavel_container.ext3` overlay on
  `/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif`; torch 2.0.1+cu117
  (CUDA-enabled). Add `--nv` only for GPU runs.
- Gotcha: some GPU nodes were pathologically slow during the RAM preload (jobs hung
  ~1 h in `read`/`load`, one CPU core pegged, GPU idle). This is slow node I/O, **not**
  a code/GPU-compute problem -- resubmit if a job stalls in preload.

## Corrections made during this work (for the record)

- "GPU hurts the skill path" was **wrong** -- it came from one anomalous stalled node.
  A clean A/B showed GPU ~neutral per-slice (~1.2x) and ~10x once batched.
- Vectorizing the per-slice *output assembly* regressed (0.6x) and was reverted -- the
  assembly was never the bottleneck; the per-slice inference + `select2d` was.
- The first "66x per-iter" was inflated by comparing a `cs`-CPU run to a GPU run on a
  faster `l40s` CPU; the confound-corrected same-hardware number is ~55x per iter.

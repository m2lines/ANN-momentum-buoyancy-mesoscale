# Channel side-plan: LINEAR-EOS truth + resolution convergence (2026-06-25)

**Purpose.** (1) Re-create the deleted LINEAR `woc_p0625` truth, and (2) run one finer
(`p03125`) wind to demonstrate that the large-scale metrics are converged at `p0625`.
Both staged as warm-starts (change one thing per run). Sits inside the bigger channel-run
program for Part 2 §4 (see PLAN.md "CHANNEL-RUN PLANNING").

**Config.** `extra_sponge_slow` + `woc` (LINEAR EOS, density-space diagnostics) — the
analysis-grade, theory-matched family. Headline wind **τ=0.2** (single wind for both stages).

**Term note.** "warm-start" = restart from the previous equilibrium, not from rest.
- Stage 1 = true restart-from-file (Z* prognostics u,v,h,T,S carry over; density is
  *diagnosed*, so flipping `EQN_OF_STATE` needs no interpolation — the state re-adjusts).
- Stage 2 = re-initialize from *interpolated* fields (different grid).

---

## Stage 0 — prep (VERIFIED 2026-06-25)
- LINEAR `woc` override block (from surviving woc coarse runs):
  `EQN_OF_STATE=LINEAR`, `LIGHTEST_DENSITY=1026.2`, `NUM_DIAG_COORDS=2` (z+rho2),
  time-mean diag_table (`prog_tmean`/`prog_z_tmean`/`prog_rho_tmean`).
  **Also enable** `CALCULATE_APE=True` (APE in ocean.stats — free; its *use* is under discussion but
  output is harmless) and register `PE_to_KE` (3D) in the diag_table (energy-conversion metric).
- **Sponge — ✅ CLEARED (was the main risk):** sponge restores **potential temperature**
  (`SPONGE_PTEMP_VAR="Temp"`), so it is EOS-independent → **reuse `sponge_damping_NI640_NJ368_relax_360_days.nc`
  as-is for LINEAR** (no regen). Stage 2 still needs a NEW `..._NI1280_NJ736_...` sponge — but since the target
  is a T field, just regrid/interpolate it.
- **Restart — ✅ verified:** WRIGHT p0625 `tau_0.2` `RESTART/` has full `MOM.res.nc` (749 MB) + `ocean_solo.res`
  at the equilibrated day-~10000 end-state → clean restart-from-file.

## Stage 1 — EOS switch @ p0625 (WRIGHT → LINEAR)
- Restart existing WRIGHT `extra_sponge_slow_p0625/tau_0.2_...` from its restart file with
  the LINEAR `woc` block. No interpolation.
- Run to re-equilibration (monitor ACC transport + domain KE); expect **~8–10 model-yr**
  (slowest adjuster = 360-day sponge / deep stratification).
- ~**5 h wall on 480 cores** (6 nodes), 1 job.
- **Output: LINEAR `woc_p0625` truth (τ=0.2)** — doubles as launch state for Stage 2.

## Stage 2 — resolution jump @ p03125 (LINEAR)
- Interpolate equilibrated LINEAR p0625 → p03125 (1280×736×30) as IC.
  `DT 900→450 s`, layout ~48×40, ~**24 nodes** (4× memory).
- Run ~**6 model-yr** (1–2 yr fine-eddy spin-up + ~3–4 yr averaging).
- ~**6–8 h wall**, 1–2 chained 16 h jobs.
- Output: LINEAR `woc_p03125` (τ=0.2).

## Stage 3 — convergence analysis
- Compare p03125 vs p0625 (both LINEAR, τ=0.2): **ACC transport, Ψ(y,ρ) max + structure,
  isotherm/thermocline depth, domain EKE.**
- Adaptive stop: small delta → converged, write it up; large delta → *then* consider p015625.

---

## Cost
| stage | model-yr | cores | wall |
|---|---|---|---|
| 1 (EOS switch p0625) | ~8–10 | 480 (6 nodes) | ~5 h |
| 2 (p03125 convergence) | ~6 | ~1920 (24 nodes) | ~6–8 h |

Baseline measured: p0625 = 27.4 model-yr in 13.3 h on 480 cores (≈233 core-h/model-yr);
p03125 ≈ 8× compute, 4× memory.

## Implementation deltas for compute-Claude (vs existing launcher)
- Restart-from-existing-run path (Stage 1) instead of cold base_exp.
- p03125 resolution preset (NI1280×NJ736) + its sponge file.
- p0625→p03125 interpolation of the IC.

## Open decisions
1. Headline wind — τ=0.2 assumed.
2. Stage-1 truth breadth — τ=0.2 only now; extend to more winds later if the online-eval
   truth needs them (don't over-build before convergence result is in).

**Status:** plan written 2026-06-25; not yet launched. Awaiting headline-wind confirm +
big-picture run-matrix decisions before handing the spec to compute-Claude.

# Channel online program — Part 2 §4 (master plan, decisions locked 2026-06-25)

The first online configuration for Part 2: does the buoyancy-flux ANN improve a
non-eddy-resolving channel toward an eddy-resolving truth, across resolution and C_ANN?
Online-eval *design* locked separately (PLAN.md "ONLINE-EVALUATION DESIGN"). This is the
concrete run matrix + diagnostics + fan-out. Convergence/truth sub-plan:
[[CHANNEL_CONVERGENCE_PLAN.md]]. B-axis toolkit sub-plan: TBD (next discussion).

## Locked decisions
| axis | choice |
|---|---|
| Wind | single, **τ=0.2** (no wind axis — that's the follow-up study, with cu) |
| Exploration axes | **resolution × C_ANN** (Part-1 style) |
| Scheme | **buoyancy ANN only (cb), cu=0**, neutral `EXP_neutral_all4` |
| C_ANN | sweep **cb ∈ {0,1,2,3,4}** (cb=0 = no-param floor) |
| EOS/config | **LINEAR `woc`** (`extra_sponge_slow`): clean material density, theory-matched |
| Resolutions | **2/3° (CESM)**, **0.25° (OM5)** headline; 0.5°, 1.0° fill the ladder; truth 1/16°; conv. 1/32° |
| Baselines | ladder: no-param / **const-GM** (ref, ≤2 κ) / **MEKE** (primary, reviewer-proof) / ANN |

## Baseline ladder — all skew-flux (GM-family) closures; differ only in how flux/κ is set
| scheme | κ / flux determined by | status |
|---|---|---|
| no-param | κ=0 | reuse existing woc cb=0 (model-agnostic) |
| const-GM | κ=const (Part-1 baseline) | NEW (≤2 κ values) |
| **MEKE** | prognostic eddy energy (energetically constrained) | NEW (in MOM6 build; cf. OM4_MEKE_GM) |
| **ANN (ours)** | learned from local state | re-run neutral (existing are σ0) |

## Run matrix — all τ=0.2, LINEAR woc, neutral ANN
Coarse (cheap, run fast). Per resolution: no-param + ANN cb{1,2,3,4} + const-GM(≤2) + MEKE.

| resolution | NI×NJ | no-param | ANN cb{1-4} neutral | const-GM | MEKE |
|---|---|---|---|---|---|
| 1.0° | 40×23 | reuse woc cb0 | re-run | new | new |
| **2/3° (CESM)** | ~60×34 **NEW preset** | new | new | new | new |
| 0.5° | 80×46 | reuse woc cb0 | re-run | new | new |
| **0.25° (OM5)** | 160×92 | reuse woc cb0 | re-run | new | new |
| truth 1/16° | 640×368 | [[CHANNEL_CONVERGENCE_PLAN.md]] Stage 1 | — | — | — |
| conv. 1/32° | 1280×736 | Stage 2 | — | — | — |

≈ 7–8 coarse runs × 4 resolutions (~30 cheap runs) + 2 expensive HR (truth + convergence).

## Diagnostics — A / B / C (streamlined 2026-06-25; APE block PROVISIONAL — see ⚠️ note)
- **A (mean state):** A1 ACC transport (total); A2 **baroclinic / thermal-wind transport** (where cb acts;
  `compute_thermal_wind_transport` exists); A3 **isopycnal slope + N², made quantitative** (mean slope over
  the resolved band, RMS bias vs truth — the direct target of a thickness closure). *Dropped:* MLD, isotherm
  depth, jet-as-metric (jet u(y,z) kept only as a context panel — transport already encodes it).
- **B (energetics/mechanism — the affirmative axis):**
  - B1 **Ψ(y,ρ) residual overturning** = realized net transport (outcome) — **supporting**, ~free given B2.
  - B2 **Ψ\* eddy-induced (bolus) streamfunction** = scheme's direct contribution. Take the scheme's **own Υ
    output** (GM κs / MEKE κ_MEKE s / **ANN predicted, NOT ∝ slope**); truth Ψ\* = residual−Eulerian
    (density-space − z-space transport, covariance-free). **NOT a scalar κ** — κ is only the down-gradient
    projection (the GM-equivalent part), erasing the ANN's contribution by construction. The **non-slope-aligned
    content** of Ψ\* (counter-gradient / structure no κ≥0 can make) is the ANN-vs-GM discriminator (absorbs the
    old "C1"). κ = one-line footnote only.
  - B3 **APE budget (PROVISIONAL ⚠️)** — three pieces of one cycle; the *same* interface-displacement Lorenz
    APE recurs across MOM6 / Pavel 2025 / Part 1 / our SI:
    · *reservoir* = **MOM6-native APE** (`ocean.stats`; exact under LINEAR EOS; = Pavel's reported APE),
      cross-checked vs the offline Part-1 fixed-reference form.
    · *scheme sink* = **(∂ₜAPE)^SF = ρ₀F·∇M** (Part 1) = our offline APE-sink ∫ρ₀F_h·∇_hρ/∂_zρ integrated
      (mean/eddy split) — closes the offline→online through-line.
    · *conversion* = **`PE_to_KE`** (3D) — the APE→EKE pathway (supports B4).
  - B4 **EKE-preservation** (eddy = deviation from zonal+time mean) — the main ANN-vs-GM win.
- Skill vs the LINEAR woc truth, in the resolved band (north of ~−40°); southern margin caveated.
- *Dropped:* C-axis "structure maps" as a line item (that's how A/B are displayed, not a metric).

⚠️ **APE definitions are PROVISIONAL — Dhruv skeptical; careful discussion REQUIRED before implementing B3.**
Agenda for that discussion: (i) the written forms aren't trivially identical — Pavel (η²−η_ref²), Part-1
si.tex ((η−η_ref)²), code ((hint²−hbot²)); confirm they reconcile. (ii) interface-height APE under
**z\*-coords with outcropping/grounding isopycnals** in the channel — validity at outcrops. (iii) large
isopycnal displacements (the channel's whole point) vs the quadratic-in-η form. (iv) **sorted** (MOM6,
time-varying) vs **fixed** (Part-1) reference — which for cross-run comparison. (v) MAPE/EAPE split + the
mean/eddy tendency decomposition. Treat the recommendation above as the *starting hypothesis*, not settled.

**B-axis toolkit (sub-plan, fns into `channel_analysis_mod`):** Ψ(y,ρ) [`prog_rho_tmean` vmo],
Ψ\* [scheme Υ direct; truth = density-space − z-space transport, covariance-free, no Fortran for truth],
APE [reservoir from `ocean.stats`; **sink + conversion pending the ⚠️ discussion**], EKE [zonal+time-mean],
A3 slope/N² [light reduction], + coarse-grain helper. The ANN's own Υ output is the shared dependency for
B2's scheme-Ψ\* + the sink term (one Fortran/diag check solves both). Each fn ships one assert self-check.

## Narrative
Total transport is momentum-dominated (expect ANN-flat) — our scheme's value is the **B-axis**
(overturning / APE / restratification) and the **baroclinic transport**. Frame accordingly:
A-axis for honesty/completeness, B-axis for the affirmative claim.

## Fan-out (two independent arms)
- **compute-Claude:** [[CHANNEL_CONVERGENCE_PLAN.md]] Stages 1→2 (truth + convergence) ∥ the coarse
  program (new preset 2/3°; neutral-ANN re-runs; const-GM; MEKE) — all τ=0.2, LINEAR woc, independent.
- **this Claude:** B-axis toolkit + A/B/C notebook against existing no-param runs (no waiting);
  §4 prose scaffold.

## Compute-Claude hand-off (ready to dispatch once toolkit-sub-plan doesn't block it — it doesn't)
1. Build **2/3° resolution preset** (NI/NJ + sponge file `..._NI60_NJ34_relax_360_days.nc`).
2. **Coarse neutral-ANN re-runs:** swap `meso_sfn_ann_file` → `EXP_neutral_all4` export; cb{1,2,3,4},
   cu=0, τ=0.2, LINEAR woc, at {1.0°, 2/3°, 0.5°, 0.25°}.
3. **const-GM** (≤2 κ) and **MEKE** coarse runs at the same 4 resolutions.
4. **HR:** [[CHANNEL_CONVERGENCE_PLAN.md]] Stage 1 (LINEAR woc_p0625 truth) then Stage 2 (woc_p03125).

## Open / next
- ⚠️ **APE careful-discussion (REQUIRED before B3 implementation)** — Dhruv skeptical of all the APE defs;
  agenda in the ⚠️ note above (form reconciliation, z\*/outcrop validity, large-displacement, reference state,
  MAPE/EAPE). The reservoir/sink/conversion triplet is a *starting hypothesis*, not locked.
- **B-axis toolkit sub-plan** (Ψ(y,ρ), Ψ\*, EKE design) — can proceed; APE (B3) waits on the discussion above.
- const-GM κ values; MEKE channel-param tuning.
- Export path for the neutral `EXP_neutral_all4` ANN netcdf for MOM6.

**Status:** decisions locked 2026-06-25; ready to fan out. APE defs provisional (careful discussion pending);
toolkit sub-plan + compute hand-off otherwise ready.

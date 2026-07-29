# NeverWorld2 (§4.2) — run plan

Drafted 2026-07-29 with Dhruv, after surveying what already exists. NW2 is the realistic-config
headline of §4; the channel (§4.1) is the controlled laboratory. Nothing here is launched yet.

## What already exists (verified, not assumed)

**Our runs** — `/scratch/db194/mom6/dec2025/NeverWorld2/`, ~750 GB, all `exitcode=0`:

| | R2 = 1/2° (120×280) | R3 = 1/3° (180×420) | R4 = 1/4° (240×560) |
|---|---|---|---|
| ANN | c1.0, c2.0, c2.5 — 30 windows (d30050–59050) | c1.0/2.0/2.5/3.5 — 2 windows | c1.0 (8), c2.5 (11), c2.5_mom (10) |
| GM  | κ=200, κ=800 — 30 windows | — | — |
| MEKE| khf=0.2 — 30 windows | — | — |

All ANN runs use `ann_instance_20Dec.nc` — the **old σ0 network**, six months before the canonical
`EXP_neutral_all4`. This is the blocking gap: §3.4 now commits to the neutral framework, so
publishing σ0 results would contradict our own methods section.

**Pavel's tree** — `/scratch/pp2681/mom6/Neverworld2/simulations/` (readable by us):
- `R32/` (925 GB) — the 1/32° truth, **already filtered+coarsened** to our grids:
  `{1,0.5,0.25}-degree-coarsen-snapshots.nc` (6–97 MB, `h,u,v`, 4 time records) plus
  `{...}-static.nc`; `{...}-degree-longmean.nc` are the same filtering on the native 1/32° grid
  (~21 GB each, includes `e`); `filter_scale_0.75/` is what his Figure 2 reads.
- `R{2,3,4}-long/bare` — **unparameterized baselines**, with restarts at day 29050
  (`MOM.res_Y0081_D201_S00000.nc`). Only 1–2 longmean windows each, so short on statistics, but
  they give us both the missing no-param baseline *and* a common equilibrated branch point.
- His published online metric (Figure S3 / Tables S2–S3) is **interface-depth RMSE vs R32** along
  fixed longitudes — a mean-state measure. He did not do the APE/EKE map comparison we built for
  the channel, so our B-axis is additive rather than duplicative.

**Binary** — `/home/db194/MOM6-examples/build/intel/ocean_only/repro/MOM6` (rebuilt 2026-07-11)
has `MESO_SFN_UPSILON_FORM`, `MESO_UPSILON_CLAMP`, `MESO_SFN_SLOPE_MAX`,
`calc_layered_density_gradients`, and **no** `meso_sfn_ann_type` ⇒ it is the bounded-Υ lineage with
UPPERCASE parameters. The dec2025 `MOM_override` files use the old lowercase names and
`meso_sfn_ann_type=nondim_rhoF_ann`; they must be rewritten.

## Two things NW2 changes relative to the channel

1. **NW2 is LAYERED** (`USE_EOS=False`, `USE_REGRIDDING=False`, 15 isopycnal layers). Density
   gradients come from `calc_layered_density_gradients`, not `calc_isoneutral_slopes`. Two
   consequences: the bounded forms' sine-form slopes were built to work in layered mode (that was a
   design requirement) but have **never been exercised there online**; and
   `MESO_SFN_MIN_DIST_BOUNDARY = 50 m` **is active in layered mode** (the guard sits inside
   `if (.not. use_EOS)`), unlike the channel where `USE_EOS=True` made it inert.
   ⇒ "no limiters" cannot mean the same thing here. Proposal: keep the layered boundary guard (it is
   structural to the layered path, not a tunable cap), disable the three internal limiters as in the
   channel, and say so explicitly in §4.2 rather than let the §4.1 sentence over-reach.
2. **Cost** (measured from `CPU_stats`, 48 PEs): R2 ≈ **48 min per 1000 days**; R4 ≈ **4.5 h per
   1000 days**. A 30 000-day production leg is ~24 h at R2 (fits the 48 h wall) but ~6 days at R4
   (needs restart-chained segments, as we did for the bounded-Υ channel runs).

## Proposed runs

**Phase 1 — R2 (1/2°), the publishable set.** Common warm start from `R2-long/bare` day 29050,
30 000 days, ~24 h each, all six independent:

| run | config |
|---|---|
| `bare` | no parameterization (extend Pavel's for statistics) |
| `GM200` | constant κ = 200 |
| `GM800` | constant κ = 800 |
| `MEKE` | `MEKE_KHTH_FAC = 0.2` |
| `ANN_c1p0` | canonical neutral model, C_ANN = 1, STENCIL_GRAD, limiters off |
| `ANN_c2p0` | as above, C_ANN = 2 (coefficient sensitivity only) |

Rationale: R2 is the only resolution where a complete closure comparison already exists, so
re-running it with the canonical network is the minimum credible §4.2 and reuses proven configs.

**Phase 2 — R4 (1/4°), only if Phase 1 lands.** `bare`, `GM` (one κ), `ANN_c1p0`, segmented.
This buys the eddy-permitting regime that made the channel's two-regime argument work. R3 is a
distraction — drop it.

**Not proposed:** re-running the σ0 sweep, or c2.5/c3.5. §4.1 already establishes that C_ANN
saturates and should not be tuned; NW2 does not need to re-litigate it, and one sensitivity run
(c2.0) suffices to show the same behaviour carries over.

## Diagnostics — write less

Current `longmean` writes `u,v,h,e,uh,vh,cg1,Rd1,Rd_dx,KE,uhGM,vhGM,GM_sfn_unlim_{x,y},GM_sfn_{x,y}`;
`snapshots` adds `RV` and `meso_sfn_drd{x,y,z}_{u,v}`. R4/ANN_c2p5 cost 157 GB for 11 windows.

What the A/B/C axes actually need:
- **longmean**: `u, v, h, e, uh, vh` — mean state, overturning, transport, and interface-displacement APE.
- **snapshots**: `u, v, h` — resolved EKE from snapshot variance.

Drop `cg1, Rd1, Rd_dx, KE` (derivable or unused), the four `GM_sfn*` fields and `uhGM/vhGM`
(debugging aids from the bounded-Υ work), `RV`, and the `meso_sfn_drd*` gradients. Estimated saving
~40–50% of write volume. Keep `static.nc`.

Note the truth products carry only `h,u,v`, so `e` for the truth must be rebuilt by cumulative sum
of `h` — the same route the channel APE estimator already uses.

## Open questions before launching

1. Does `STENCIL_GRAD` behave in **layered** mode? Needs a short smoke test before committing
   30 000-day legs — the bounded forms have only ever run online in EOS mode.
2. Truth sampling: the coarsened snapshots have only 4 time records, which is thin for an EKE
   estimate. Either coarsen `0.5-degree-longmean.nc` ourselves (21 GB, has `e`) or ask Pavel whether
   a longer coarsened record exists.
3. Confirm with Pavel that `R{2,4}-long/bare` day-29050 restarts are the state his own runs branched
   from, so our baseline and closure runs share a common history.

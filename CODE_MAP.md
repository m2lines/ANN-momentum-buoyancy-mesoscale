# CODE_MAP — training pipeline reference

Navigable map of `src/training-on-CM2.6/` (the Python training/eval code), produced from a
full code audit on 2026-06-16. Line numbers are accurate as of that date but **shift when files
are edited** — trust the structure and function names over exact lines. Companion docs:
`PLAN.md` (state of record), `DB_notebooks/INVENTORY.md` (notebooks), `src/README.md` (MOM6 patch).

The repo extends **Pavel Perezhogin's momentum pipeline** to **buoyancy/ρ fluxes**. Much of the
"craziness" is upstream momentum machinery; the ρ extension reuses it with targeted swaps.

---

## 1. Big picture: data flow

```
download_raw_data.py ──► /vast/$USER/CM26_datasets/ocean3d/rawdata/{train,validate,test}-*.nc
   (raw high-res CM2.6,                  (test-0.nc … = 5.5 GB each, global 1/10°, 3600×2700×50)
    splits: train yrs181-188=96 snaps,
    validate yr194=12, test yrs199-200=24)
        │
generate_3d_datasets.py --factor=N      (filter + coarsen; --add_rho_fluxes defaults to 1)
        ▼
/vast/$USER/CM26_datasets/ocean3d/{subfilter}/FGR{FGR}/factor-{N}/
   param.nc, permanent_features.nc, {train,validate,test}-*.nc   ← has Fx,Fy + all ANN inputs
        │
read_datasets()  ──►  DatasetCM26  ──►  three parallel training entry points (see §2)
                                          └► predict_* → SGS_skill* → skill-test/factor-N.nc
```

**Where data actually lives (2026-06-16):** the ρ truth `Fx`/`Fy` only exists in the
`add_rho_fluxes` regeneration on **`/vast/db194`** (Dhruv's). Pavel's reachable copy on
**`/scratch/pp2681/CM26_datasets`** is momentum-only (has inputs, no `Fx`/`Fy`), but its
**`ocean3d/rawdata`** IS the raw high-res data, so ρ data can be regenerated from it (see PLAN).
**Env to run any of this:** the `Pavel_Container` Jupyter kernel (`/scratch/$USER/Pavel_container.ext3`,
torch/xarray/xgcm/gcm_filters; no jax — jax is training-only). Inference is torch, runs on CPU.

---

## 2. The THREE parallel pipelines (the key structure)

| | **momentum forcing** | **momentum flux** | **ρ flux (this fork)** |
|---|---|---|---|
| script | `train_script.py` | `train_script_fluxes.py` | `train_script_rho_fluxes.py` |
| loop | `train_ann.py:train_ANN` | `train_ann_fluxes.py:train_ANN_fluxes` | `train_rho_fluxes.py:train_ANN_rho_fluxes` |
| target | SGSx/SGSy via ZB20u/v | Txx,Tyy,Txy | **Fx,Fy** |
| inference fn | `Apply_ANN`/`ANN` | `ANN_inference` | `ANN_rho_inference` |
| ANN outputs | 3 | 3 | **2** |
| inputs | gradient_features (sh_xy,sh_xx,rel_vort) | same 3 | **5: rhox,rhoy,sh_xx,sh_xy,rel_vort** (hardcoded) |
| symmetry aug | yes (rot/reflect) | yes | **NONE** (move-fast simplification) |
| dimensional_scaling | flag (True) | always | always (rho_norm·gradv_norm·Δ²) |
| data fetch | `select2d` (no polar-fold drop) | `fetch_data` (drops top 2 `yh`) | `select2d`+`drop_polar_fold` (drops top 2 `yh`+`yq`) |
| depth_idx default | `arange(10)` (10 lvls) | `arange(10)` | **`arange(0,50,2)` (25 lvls)** |
| save path | `…/CM26_ML_models/ocean3d/{subfilter}/FGR{FGR}/{exp}/model/Tall.nc` | same | **`…/CM26_ML_models/FGR{FGR}/{exp}/model/ann_instance.nc`** (no `ocean3d/`!) |
| skill fn | `SGS_skill` | `SGS_skill` | **`SGS_skill_rho`** |

⚠️ The ρ save path is **different** (no `ocean3d/{subfilter}/`, file `ann_instance.nc` not `Tall.nc`).
That's why the canonical model lives at `…/FGR3/EXP0/…/ann_instance_20Dec.nc` (and was committed
to `CM26_ML_models/ocean3d/subfilter/FGR3/buoyancy/hidden-layer-32-32/…` — a third, hand-placed path).

---

## 3. Module reference

### `cm26.py` (1007 lines) — `DatasetCM26` + dataset I/O
- `read_datasets()` (~L12) — load coarse datasets. **Hardcoded `/vast/$USER/CM26_datasets` (L16).**
- `create_grid()` (~L71) — xgcm Grid (X periodic, Y fill). 2D/3D.
- `propagate_mask()`, `discard_land()` — coastal mask dilation (percentile rule).
- `from_cloud()` (~L94) — raw-data loader. cloud (`leap`/`cmip6`/`cmip6-3d` zarr) **or** local
  `3d-*` from **hardcoded `/vast/pp2681/.../rawdata` (L118)** ← documented config point (README L73).
- `select2d()` (~L248) — single time+depth slice (random if None). **No polar-fold drop.**
- `init_coarse_grid()`, `coarsen()` — build coarse grid; filter (FGR) then box-coarsen.
- `compute_subfilter_forcing()` (~L426, momentum) / `add_subfilter_forcing_rho()` (~L543, ρ).
  **Sign: `F = bar()bar() − bar(()())  = −(subfilter)`** (uncentered). ρ flux `Fx=ρ̄ū−ρu‾`.
  ρ version interpolates u,v to tracer points *before* filtering (all vars co-located).
- `predict_ANN()` (~L613, momentum, whole-dataset) / `predict_ANN_rho()` (~L651, ρ, **implemented 2026-06-16**) / `predict_ZB()` (~L692, ZB20 baseline).
- `SGS_skill()` (~L759, momentum) — defines nested `M2`/`M2u`/`M2v` second-moment helpers;
  R2T/corr_T (map/centered/away/lon variants), dissipation, transfer/power spectra over
  regions NA/Pacific/Equator/ACC, dEdt. `SGS_skill_rho()` (~L676, **added 2026-06-16**) — R2F/corr_F for Fx,Fy.
- `nanvar()` — mask to wet at the right grid stagger.

### `state_functions.py` (2345 lines) — `StateFunctions` (physics + ML features)
The big one. Themes:
- **Density/stratification**: `rho(potential=True)` (~L1774, σ0 via gsw; in-situ branch assumes p≈z, valid <500 m), `Nsquared` (~L1805, hardcoded g=9.8, ρ0=1025, clipped ≥0), `baroclinic_speed`.
- **Velocity gradients**: `velocity_gradients()` (~L703) — **sign conventions: `sh_xx=dudx−dvdy`, `sh_xy=dvdx+dudy`, `vort_xy=dvdx−dudy`, `div=dudx+dvdy`**. `relative_vorticity()` (~L1663, MOM6 corner formula). `KE_Arakawa`, `gradKE`.
- **Geostrophic/instability features**: `vertical_shear_geostrophic()` (~L1931) — thermal wind `du/dz=−g/(ρ0 f)∂yρ`, `dv/dz=+g/(ρ0 f)∂xρ`; **also produces the `rhox/rhoy` ANN inputs** (gradients of `self.data.rho` = σ0). `Eady_time*`, `deformation_radius` (~L1857, blended f/β formula).
- **ANN application**:
  - `Apply_ANN` (~L1090) / `ANN` (~L1433): momentum Txx/Tyy/Txy + ZB20u/v; dimensional or flat (1e-6) input norm; **symmetry rotation/reflection via `sign_mapping`**; **minus sign applied to momentum outputs** (verify against ZB ref).
  - `ANN_inference` (~L1469): faster momentum variant (Txx/Tyy/Txy only).
  - `ANN_rho_inference` (~L1574): **ρ — predicts Fx,Fy.** Inputs rhox,rhoy,sh_xx,sh_xy_h,rel_vort_h; norm by Frobenius (rho_norm, gradv_norm); output `× rho_norm·gradv_norm·(wet·Δ²)`, Δ=√(dxT·dyT). **No rotation/symmetry.**
- **Feature prep**: `compute_features()` (~L1041, `@lru_cache`) — builds sh_xy/sh_xx/vort + `_h`/`_q` interpolations. `prepare_features()` (~L2108) — the 3D features stored to disk (N, Rd, rescaled_depth z_s, dudz, dudz_geo, rhox, rhoy, SGS_KE, Eady Te, deltaU, …).
- **Closures/diagnostics**: `ZB20()` (~L862, Zanna-Bolton with VGM variants), `Smagorinsky*`, `JansenHeld` backscatter, `advection`/`PV_cross_uv`, `vertical_modes*`.
- Hardcoded constants: g=9.8, ρ0=1025, Ω=7.2921e-5, Re=6.371e6, strain_norm=1e-6, flux_norm=1e-2.

### `ann_tools.py` (347 lines) — the ANN + MOM6 netcdf format
- `ANN` class (~L110) — feedforward MLP, ReLU hidden, linear out; `layer_sizes`.
- `export_ANN`/`import_ANN` (~L135/176) — **the netcdf MOM6 reads**: `num_layers`, `layer_sizes`,
  weights `A{i}` (transposed), bias `b{i}`, `x_test`/`y_test` (round-trip check), `input_norms`/`output_norms`.
- `tensor_from_xarray`, `torch_pad` (**circular zonal**, zero meridional), `image_to_nxn_stencil_gpt`
  (stencil extraction via `unfold`, supports rotation 0/90/180/270 + reflections — this is where
  the momentum symmetry augmentation happens; ρ doesn't use it).

### `operators.py` (216 lines) — coarsen/filter
- `CoarsenWeighted` (area-weighted), `CoarsenKochkov`/`CoarsenKochkovMinMax` (divergence-preserving;
  **MinMax is the generation default**), `Subsampling`, `Filtering` (gcm_filters, GAUSSIAN default,
  `filter_scale=FGR`, REGULAR_WITH_LAND, per-field wet masks).

### `feature_extractors.py` (141 lines) — scalar state features
- `grid_step` (÷50 km), `deformation_radius` (+linear/over-grid variants), `square_root_of_Ri`,
  `Held_Larichev_1996` (+linear/inverse variants), `rescaled_depth`. Each has **hardcoded log/linear
  clip ranges** (empirically tuned). Return `(corner, center)` tuples.

### `selectors.py` (188 lines) — regions + `compare()`
- Region selectors: `select_globe/NA/Pacific/Equator/ACC/SO/Gulf/Kuroshio/Aghulas/Malvinas/...` (lat/lon boxes).
- `compare()` (~L135) — side-by-side plot with embedded **R² (uncentered), corr (centered), optimal_scaling**.
  This is what `offline_eval.ipynb` uses for qualitative ρ maps.

### `plot_helpers.py` (267 lines) — figures/animation
- `create_animation*`, `merge/split_gif`, `default_rcParams`, `imshow`, `set_letters`. No skill math here.

### scripts/
- `download_raw_data.py` — fetch raw CM2.6 (path L11). `generate_3d_datasets.py` — filter+coarsen per
  factor (`--add_rho_fluxes` default **1**, `CoarsenKochkovMinMax`, `Filtering`). `slurm_generate_datasets_{N}.sh`
  (14 cpu, 128 GB, 24–60 h, Pavel_container), `slurm_train_ann.sh` (runs the ρ trainer: `[32,32]`, 500 iters, EXP0),
  `launcher.sh` (dev).

---

## 4. Critical conventions (don't get these wrong)
- **Subfilter flux sign is NEGATED**: stored `F = bar()bar() − bar(()()) = −(physical subfilter)`.
  The MOM6 Fortran negates back (`Fx_c = -yy`). Verified consistent (PLAN M1).
- **Gradient signs**: `sh_xx=dudx−dvdy`, `sh_xy=dvdx+dudy`, `vort=dvdx−dudy`. Match the Fortran.
- **Non-dim (ρ)**: inputs ÷ Frobenius norm (range-limited); output × `Δ²·|∇ρ|·|∇u|`. Same as Part 1 with thickness→density.
- **Offline density = potential σ0** everywhere (flux target + rhox/rhoy inputs). Online uses isoneutral
  (locally-referenced) density → known train/online mismatch (PLAN).

## 5. Hardcoded paths / config points
| file:line | path | note |
|---|---|---|
| `cm26.py:16` | `/vast/$USER/CM26_datasets/...` | coarse dataset read path |
| `cm26.py:118` | `/vast/pp2681/.../rawdata` | raw input; **documented config point** (README) |
| `download_raw_data.py:11` | `/vast/$USER/.../rawdata` | raw output |
| `generate_3d_datasets.py:29-30` | `/vast/$USER/CM26_datasets/ocean3d` | coarse output |
| train scripts | `/scratch/$USER/mom6/CM26_ML_models/...` | model+skill output (ρ path differs, §2) |

## 6. Known bugs / gotchas / inconsistencies
- **FIXED 2026-06-16**: `predict_ANN_rho` stub → implemented; `SGS_skill_rho` added; `train_script_rho_fluxes`
  testing block rewired (removed `args.gradient_features` AttributeError); `drop_polar_fold` added to ρ trainer.
- **Open**: σ0-vs-isoneutral density mismatch (online); ρ dropped Part-1's flow-aligned rotation/symmetry
  (move-fast simplification — decide keep-vs-retrain from offline skill); `cm26.py` hardcoded paths.
- Inconsistency: ρ model save path/filename differs from momentum (§2). Depth defaults differ (10 vs 25 lvls).
- `train_ann.py` has a **dual-loss** path (forcing vs fluxes) — `train_script.py` hardcodes `loss_function='forcing'`.
- Agent-flagged, **to verify** (lower confidence): minus-sign provenance on momentum ANN outputs;
  `Apply_ANN` Txx/Tyy shape handling when `dimensional_scaling=False`; `vertical_modes` dead code after early return.

## 7. What this fork changed vs upstream momentum
- Added: `add_subfilter_forcing_rho`, `ANN_rho_inference`, `train_rho_fluxes.py`, `train_script_rho_fluxes.py`,
  `predict_ANN_rho` + `SGS_skill_rho`, `--add_rho_fluxes` flag, `DB_notebooks/` analyses, the buoyancy paper.
- Reused unchanged: the whole feature/operator/ANN/skill machinery, just retargeted T→F (3→2 components).

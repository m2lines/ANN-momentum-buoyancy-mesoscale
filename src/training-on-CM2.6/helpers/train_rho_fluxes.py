import sys
import numpy as np
import xarray as xr
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import ANN, tensor_from_xarray
import torch
import torch.optim as optim
import itertools

import os
from time import time

def get_rho_fluxes(batch, device='cpu'):
    Fx = tensor_from_xarray(batch.data.Fx).to(device)
    Fy = tensor_from_xarray(batch.data.Fy).to(device)

    F_norm = 1. / torch.sqrt((Fx**2 + Fy**2).mean())
    Fx = Fx * F_norm
    Fy = Fy * F_norm

    return Fx, Fy, F_norm

def torch_flux_div(Fx, Fy, dyCu, dxCv, inv_area, wet):
    """Differentiable C-grid flux-form divergence of a vector flux at cell centres, the
    buoyancy analog of the momentum stress divergence (ZB20u/v): interp the flux to the
    cell faces, difference, divide by area. Periodic in X; the Y edges wrap but are masked
    by wet (the polar fold is dropped upstream). Fx,Fy at centre; dyCu at the u-face,
    dxCv at the v-face; inv_area = 1/(dxT*dyT)."""
    Fx = torch.nan_to_num(Fx); Fy = torch.nan_to_num(Fy)
    flux_x = 0.5 * (Fx + torch.roll(Fx, -1, dims=-1)) * dyCu      # east-face (xq) flux, periodic X
    flux_y = 0.5 * (Fy + torch.roll(Fy, -1, dims=-2)) * dxCv      # north-face (yq) flux
    div = ((flux_x - torch.roll(flux_x, 1, dims=-1))
           + (flux_y - torch.roll(flux_y, 1, dims=-2))) * inv_area
    return div * wet

def drop_polar_fold(batch):
    # Drop the two northernmost rows near the Polar Fold, where fluxes and
    # B.C. are not well defined (cf. fetch_data in train_ann_fluxes).
    return DatasetCM26(batch.data.isel(yh=slice(None, -2)),
                       batch.param.isel(yh=slice(None, -2), yq=slice(None, -2)))

def train_ANN_rho_fluxes(factors=[9],
              stencil_size = 3,
              hidden_layers=[32,32],
              time_iters=50,
              learning_rate = 1e-3,
              depth_idx=np.arange(1),
              print_iters=1, 
              permute_factors_and_depth=True,
              subfilter='subfilter',
              FGR=3,
              validate_every=10,
              device='cpu',
              rotated=False,
              loss='mse',
              rho_grad_source='prod'):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors, subfilter=subfilter, FGR=FGR)

    ########## Preload into RAM ###########
    # The inner loop re-reads 2D slices off disk every step (tensor_from_xarray ->
    # to_numpy on a lazy array), which dominates runtime. Load the variables the
    # training step actually touches into memory once, so every step is RAM-fast.
    # Only these 8 data variables are used (wet comes from param, already loaded).
    # The original lazy (no-preload, validate-every-step) loop is at git 772c23b
    # (verified bit-identical weights, max abs diff 0.0); see PERFORMANCE.md.
    #
    # This holds ~130GB in RAM (8 of the file's 42 vars, train+validate, all 50
    # depths). If memory ever becomes the limit (more factors/snapshots/larger grids),
    # the easy win is to also subset depth here -- training only uses depth_idx
    # (the even levels, ~25 of 50), so `.isel(zl=depth_idx)` would roughly halve it.
    # That needs remapping the loop's `depth` index (currently the raw zl position)
    # to a position within the subset; left as full-depth here for index simplicity.
    load_vars = ['Fx', 'Fy', 'rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']
    # Pilot: optionally swap the density-gradient INPUT (rhox/rhoy) for a different "kind" --
    # 'sigma0c' (sigma0 from coarse T/S) or 'neutral' (local interface pressure, the online
    # analog). The variant fields live in factor-{f}-rhokinds (append_rho_kinds.py); same grid
    # and snapshots, so we positionally overwrite rhox/rhoy. 'prod' (default) leaves the
    # existing coarse-grained-sigma0 gradient untouched. The Fx/Fy target is unchanged.
    rk_vars = {'sigma0c': ('rhox_sigma0c', 'rhoy_sigma0c'),
               'neutral': ('rhox_neutral', 'rhoy_neutral')}
    root = os.path.expandvars(os.environ.get('CM26_DATA_ROOT', '/scratch/$USER/CM26_datasets/ocean3d'))
    for key in ['train', 'validate']:
        for factor in factors:
            d = dataset[f'{key}-{factor}']
            data = d.data[load_vars].load()
            if rho_grad_source != 'prod':
                gx, gy = rk_vars[rho_grad_source]
                rk = xr.open_mfdataset(f'{root}/{subfilter}/FGR{FGR}/factor-{factor}-rhokinds/{key}-*.nc',
                                       combine='nested', concat_dim='time').sortby('time')[[gx, gy]].load()
                assert rk.sizes['time'] == data.sizes['time'], f'{key}-{factor}: rho-kinds time mismatch'
                data['rhox'] = (data['rhox'].dims, rk[gx].transpose(*data['rhox'].dims).values)
                data['rhoy'] = (data['rhoy'].dims, rk[gy].transpose(*data['rhoy'].dims).values)
            dataset[f'{key}-{factor}'] = DatasetCM26(data, d.param)

    ########## Init logger ###########
    logger = xr.Dataset()
    logger['MSE_train'] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]),
                                       dims=['iter', 'factor', 'depth'],
                                       coords={'factor': factors, 'depth': depth_idx})
    # Validation is throttled (every 10 iters), so leave un-validated iters as NaN.
    logger['MSE_validate'] = xr.full_like(logger['MSE_train'], np.nan)

    ########## Init ANN ##############
    # 5 input features per stencil point: rhox, rhoy, sh_xx, sh_xy, rel_vort
    num_input_features = stencil_size**2 * 5
    ann_instance = ANN([num_input_features, *hidden_layers, 2]).to(device)
    
    ########## Random sampling of depth and factors #######
    def iterator(x,y):
        # Product of two 1D iterators
        x_prod = np.repeat(x,len(y))
        y_prod = np.tile(y,len(x))
        xy_prod = np.vstack([x_prod,y_prod]).T
        if permute_factors_and_depth:
            # Randomly permuting iterator along common dimension
            return np.random.permutation(xy_prod)
        else:
            # This is equivalent to
            # for xx in x:
            #    for yy in y:
            #       ....
            return xy_prod
    
    ############ Init optimizer ##############
    all_parameters = ann_instance.parameters()
    optimizer = optim.Adam(all_parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=[int(time_iters/2), int(time_iters*3/4), int(time_iters*7/8)], gamma=0.1)

    # loss on the (per-slice flux-normalized) prediction vs truth: mean-squared ('mse',
    # the default) or mean-absolute ('mae', as in Part 1). The offline R2 metric is the
    # same regardless of which loss the model was trained with.
    def loss_fn(ax, ay, fx, fy):
        if loss == 'mae':
            return (torch.abs(ax - fx) + torch.abs(ay - fy)).mean()
        return ((ax - fx)**2 + (ay - fy)**2).mean()

    # loss=='div' trains the predicted flux against the (per-slice normalized) DIVERGENCE of the
    # true flux -- the buoyancy analog of Perezhogin's momentum forcing loss. NOTE (see PLAN.md):
    # our online scheme applies the Ferrari streamfunction built from the flux, not the flux
    # divergence, so this is a comparison model, not necessarily the right objective for buoyancy.
    def compute_loss(batch, prediction):
        if loss == 'div':
            Fxt = tensor_from_xarray(batch.data.Fx).to(device)
            Fyt = tensor_from_xarray(batch.data.Fy).to(device)
            dyCu = tensor_from_xarray(batch.param.dyCu).to(device)
            dxCv = tensor_from_xarray(batch.param.dxCv).to(device)
            inv_area = (1. / tensor_from_xarray(batch.param.dxT * batch.param.dyT)).to(device)
            wet = tensor_from_xarray(batch.param.wet).to(device)
            dT = torch_flux_div(Fxt, Fyt, dyCu, dxCv, inv_area, wet)
            dP = torch_flux_div(prediction['Fx'], prediction['Fy'], dyCu, dxCv, inv_area, wet)
            n = wet.sum().clamp(min=1)
            ms_true = (dT ** 2 * wet).sum() / n                      # per-slice mean-square true div
            return ((dP - dT) ** 2 * wet).sum() / (n * ms_true.clamp(min=1e-30))
        Fx, Fy, F_norm = get_rho_fluxes(batch, device=device)
        return loss_fn(prediction['Fx'] * F_norm, prediction['Fy'] * F_norm, Fx, Fy)

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()
        # Validate only every validate_every iters (and on the last) -- a full validation
        # pass every step roughly doubles the cost for a densely-sampled curve we don't
        # need. validate_every=1 reproduces the original every-step behaviour.
        do_validate = (time_iter % validate_every == 0) or (time_iter == time_iters - 1)

        for factor, depth in iterator(factors, depth_idx):
            # Note here we randomly sample time moment 
            # for every combination of factor and depth
            # So, consequetive snapshots are not correlated (on average)
            # Batch is a dataset consisting of one 2D slice of data
            batch = drop_polar_fold(dataset[f'train-{factor}'].select2d(zl=depth))

            ############## Training step ###############
            optimizer.zero_grad()
            prediction = batch.state.ANN_rho_inference(ann_instance, stencil_size=stencil_size, device=device, rotated=rotated)
            MSE_train = compute_loss(batch, prediction)

            MSE_train.backward()
            optimizer.step()

            del batch

            ########### Logging (train every step) ############
            MSE_train = float(MSE_train.data)
            logger['MSE_train'].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = MSE_train

            ############ Validation step (throttled) ##################
            MSE_validate = None
            if do_validate:
                batch = drop_polar_fold(dataset[f'validate-{factor}'].select2d(zl=depth))

                with torch.no_grad():
                    prediction = batch.state.ANN_rho_inference(ann_instance, stencil_size=stencil_size, device=device, rotated=rotated)
                    MSE_validate = float(compute_loss(batch, prediction).data)
                logger['MSE_validate'].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = MSE_validate

                del batch

            if (time_iter+1) % print_iters == 0:
                msg = f'Factor: {factor}, depth: {depth}, MSE train: %.6f' % MSE_train
                if MSE_validate is not None:
                    msg += ', validate: %.6f' % MSE_validate
                print(msg)
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()

    for factor in factors:
        for train_str in ['train', 'validate']:
            del dataset[f'{train_str}-{factor}']
    
    return ann_instance, logger
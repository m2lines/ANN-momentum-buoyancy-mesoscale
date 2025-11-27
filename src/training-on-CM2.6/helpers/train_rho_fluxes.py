import sys
import numpy as np
import xarray as xr
from helpers.cm26 import read_datasets
from helpers.ann_tools import ANN, tensor_from_xarray
import torch
import torch.optim as optim
import itertools

import os
from time import time

def get_rho_fluxes(batch):
    Fx = tensor_from_xarray(batch.data.Fx)
    Fy = tensor_from_xarray(batch.data.Fy)

    F_norm = 1. / torch.sqrt((Fx**2 + Fy**2).mean())
    Fx = Fx * F_norm
    Fy = Fy * F_norm

    return Fx, Fy, F_norm

def train_ANN_rho_fluxes(factors=[9],
              stencil_size = 3,
              hidden_layers=[32,32],
              time_iters=50,
              learning_rate = 1e-3,
              depth_idx=np.arange(1),
              print_iters=1, 
              permute_factors_and_depth=True,
              subfilter='subfilter',
              FGR=3):
    '''
    time_iters is the number of time snaphots
    randomly sampled for each factor and depth

    depth_idx is the indices of the vertical layers which
    participate in training process
    '''
    ########### Read dataset ############
    dataset = read_datasets(['train', 'validate'], factors, subfilter=subfilter, FGR=FGR)

    ########## Init logger ###########
    logger = xr.Dataset()
    for key in ['MSE_train', 'MSE_validate']:
        logger[key] = xr.DataArray(np.zeros([time_iters, len(factors), len(depth_idx)]), 
                                   dims=['iter', 'factor', 'depth'], 
                                   coords={'factor': factors, 'depth': depth_idx})

    ########## Init ANN ##############
    # As default we have 3 input features on a stencil: D, D_hat and vorticity
    num_input_features = stencil_size**2 * 5
    ann_instance = ANN([num_input_features, *hidden_layers, 2])
    
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

    t_s = time()
    for time_iter in range(time_iters):
        t_e = time()

        for factor, depth in iterator(factors, depth_idx):
            # Note here we randomly sample time moment 
            # for every combination of factor and depth
            # So, consequetive snapshots are not correlated (on average)
            # Batch is a dataset consisting of one 2D slice of data
            batch = dataset[f'train-{factor}'].select2d(zl=depth)

            ############## Training step ###############
            Fx, Fy, F_norm = get_rho_fluxes(batch)
            
            optimizer.zero_grad()
            prediction = batch.state.ANN_rho_inference(ann_instance, stencil_size=stencil_size)
            ANNx = prediction['Fx'] * F_norm
            ANNy = prediction['Fy'] * F_norm
            MSE_train = ((ANNx-Fx)**2 + (ANNy-Fy)**2).mean()

            MSE_train.backward()
            optimizer.step()

            del batch

            ############ Validation step ##################
            batch = dataset[f'validate-{factor}'].select2d(zl=depth)
            
            Fx, Fy, F_norm = get_rho_fluxes(batch)

            with torch.no_grad():
                prediction = batch.state.ANN_rho_inference(ann_instance, stencil_size=stencil_size)

            ANNx = prediction['Fx'] * F_norm
            ANNy = prediction['Fy'] * F_norm
            MSE_validate = ((ANNx-Fx)**2 + (ANNy-Fy)**2).mean()
                
            del batch
        
            ########### Logging ############
            MSE_train = float(MSE_train.data)
            MSE_validate = float(MSE_validate.data)

            for key in ['MSE_train', 'MSE_validate']:
                logger[key].loc[{'iter': time_iter, 'factor': factor, 'depth': depth}] = eval(key)
            if (time_iter+1) % print_iters == 0:
                print(f'Factor: {factor}, depth: {depth}, '+'MSE train/validate: [%.6f, %.6f]' % (MSE_train, MSE_validate))
        t = time()
        if (time_iter+1) % print_iters == 0:
            print(f'Iter/num_iters [{time_iter+1}/{time_iters}]. Iter time/Remaining time in seconds: [%.2f/%.1f]' % (t-t_e, (t-t_s)*(time_iters/(time_iter+1)-1)))
        scheduler.step()

    for factor in factors:
        for train_str in ['train', 'validate']:
            del dataset[f'{train_str}-{factor}']
    
    return ann_instance, logger
import sys
sys.path.append('../')
import numpy as np
import xarray as xr
import torch
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.train_ann_fluxes import train_ANN_fluxes
from helpers.train_rho_fluxes import *
from helpers.feature_extractors import *
from helpers.ann_tools import ANN, export_ANN
import json
import gc

import os
import argparse

if __name__ == '__main__':
    ########## Manual input of parameters ###############
    parser = argparse.ArgumentParser()
    parser.add_argument('--stencil_size', type=int, default=3)
    parser.add_argument('--hidden_layers', type=str, default='[20]')
    #parser.add_argument('--gradient_features', type=str, default="['sh_xy', 'sh_xx', 'rel_vort']")
    parser.add_argument('--subfilter', type=str, default='subfilter')
    parser.add_argument('--FGR', type=int, default=3)
    parser.add_argument('--factors', type=str, default='[4,9,12,15]')
    parser.add_argument('--depth_idx', type=str, default='np.arange(0, 50, 2)')
    #parser.add_argument('--symmetries', type=str, default='All')
    parser.add_argument('--time_iters', type=int, default=400)
    parser.add_argument('--print_iters', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--permute_factors_and_depth', type=str, default='True')
    parser.add_argument('--validate_every', type=int, default=10,
                        help='Compute/log validation MSE every Nth iter (1 = every iter, '
                             'the original behaviour). Throttling speeds up training.')
    parser.add_argument('--seed', type=int, default=0,
                        help='RNG seed for reproducibility. Seeds numpy (factor/depth '
                             'permutation + random time sampling in select2d) and torch '
                             '(ANN weight init). Set distinct seeds for an ensemble.')
    parser.add_argument('--device', type=str, default='cpu',
                        help="'cpu' or 'cuda'. cuda runs the feature build + ANN on GPU.")

    parser.add_argument('--path_save', type=str, default='EXP0')

    args = parser.parse_args()

    # Seed before any sampling or weight init so the run is reproducible.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    path_save = os.path.expandvars(f'/scratch/$USER/mom6/CM26_ML_models/FGR{args.FGR}/{args.path_save}')

    os.system(f'mkdir -p {path_save}/model')

    print(args, '\n')
    with open(f'{path_save}/configuration.txt', "w") as outfile: 
        json.dump(vars(args), outfile)

    args.factors = eval(args.factors)
    args.hidden_layers = eval(args.hidden_layers)
    args.depth_idx = eval(args.depth_idx)
    #args.gradient_features = eval(args.gradient_features)
    args.permute_factors_and_depth = eval(args.permute_factors_and_depth)
    
    ann_Tall, logger = \
        train_ANN_rho_fluxes(args.factors,
                  args.stencil_size,
                  args.hidden_layers,
                  #args.symmetries,
                  args.time_iters,
                  args.learning_rate,
                  args.depth_idx,
                  args.print_iters,
                  #args.gradient_features,
                  args.permute_factors_and_depth,
                  args.subfilter,
                  args.FGR,
                  args.validate_every,
                  args.device
                  )

    # Move to CPU for export + the offline test. The test runs on CPU even for a
    # GPU training run: predict_ANN_rho is bound by per-slice overhead (select2d +
    # inference per 2D slice), so GPU gives little benefit (~1.2x in an A/B) -- not
    # worth a GPU allocation. The preload below is what actually speeds the test
    # up, by removing the per-slice disk reads.
    ann_Tall = ann_Tall.to('cpu')

    nfeatures = ann_Tall.layer_sizes[0]
    export_ANN(ann_Tall, input_norms=torch.ones(nfeatures), output_norms=torch.ones(2),
            filename=f'{path_save}/model/ann_instance.nc')

    logger.to_netcdf(f'{path_save}/model/logger.nc')

    LOAD_VARS = ['Fx', 'Fy', 'rhox', 'rhoy', 'sh_xx', 'sh_xy_h', 'rel_vort_h', 'delta_x']
    ds = read_datasets(['test'], [4,9,12,15], subfilter=args.subfilter, FGR=args.FGR)
    os.system(f'mkdir -p {path_save}/skill-test')
    for factor in [4,9,12,15]:
        d = ds[f'test-{factor}']
        d = DatasetCM26(d.data[LOAD_VARS].load(), d.param)   # preload -> no per-slice disk reads
        skill = d.predict_ANN_rho(ann_Tall, stencil_size=args.stencil_size).SGS_skill_rho()
        skill.to_netcdf(f'{path_save}/skill-test/factor-{factor}.nc')
        del skill
        gc.collect()
        print(f'Testing on dataset with factor {factor} is complete')

## Buoyancy ANN — density-flux parameterization

Trained ANN weights for predicting the coarse-grained eddy density flux from CM2.6 high-resolution ocean simulations. Used by the MOM6 `USE_THICKNESS_DIFFUSE_ANN` code path (module `MOM_meso_sfn_ANN.F90`) to provide a neural-network alternative to the Gent–McWilliams thickness-diffusion eddy streamfunction.

This is the buoyancy/density-flux counterpart to the momentum ANN weights stored in the sibling `hidden-layer-{20,32-32}/` directories under `subfilter/FGR3/`. It extends the 2-layer formulation of Balwada et al. (https://essopenarchive.org/doi/full/10.22541/essoar.174835313.30541637/v1) to arbitrary vertical grids by directly predicting the density flux and converting it to a streamfunction via division by the local vertical density gradient.

Architecture: stencil 3, hidden layers [32, 32], FGR=3 subfilter training. See `hidden-layer-32-32/seed-default/configuration.txt` for full training config.

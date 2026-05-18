# ANN parameterization of mesoscale momentum and buoyancy fluxes in ocean model MOM6

This repository is a fork of [m2lines/ANN-momentum-mesoscale](https://github.com/m2lines/ANN-momentum-mesoscale) that extends Pavel Perezhogin's training pipeline from momentum fluxes to buoyancy (ρ) fluxes. The buoyancy extension is developed by Dhruv Balwada with Pavel Perezhogin's help. The original momentum-flux work — paper figures, MOM6 implementation, online experiments — is unchanged from upstream and described below.

---

This repository contains training algorithm and MOM6 ocean model with implemented parameterization. Additionally, we include figure plotting notebooks for the paper "[Generalizable neural-network parameterization of mesoscale eddies in idealized and global ocean models](https://arxiv.org/abs/2505.08900)" by Pavel Perezhogin, Laure Zanna and Alistair Adcroft, to be submitted soon.

Example of online simulation with the proposed ANN parameterization in the idealized configuration NeverWorld2 of the MOM6 ocean model ([full movie](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-mp4.mp4), [code](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-movie.ipynb)):

<div align="center">
  <img src="https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/NW2-gif.gif" alt="Example GIF" />
</div>

Online simulation in global ocean model OM4 ([full movie](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-Atlantic.mp4), [code](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-movie.ipynb)):
<div align="center">
  <img src="https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/assets/OM4-gif.gif" alt="Example GIF" />
</div>

## Paper figures
In folder [notebooks](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/notebooks) we show Jupyter notebooks for each Figure plotted in the paper.

## Data
Training data, offline and online skill can be found on [Zenodo](https://zenodo.org/records/15328410).
## Installation
```
git clone --recursive git@github.com:m2lines/ANN-momentum-mesoscale.git
```

# MOM6 online experiments
* The submodule [src/MOM6](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/src) contains the ocean model source code with [implemented ANN parameterization](https://github.com/m2lines/MOM6/blob/89f1fb391d05d3f52549e4f74c74a4b4d6c01960/src/parameterizations/lateral/MOM_Zanna_Bolton.F90#L661). No additional software is required. See [src/README.md](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/README.md) to visualize modifications to MOM6 source code.
* Weights of trained ANN parameterization used for online simulations are in [CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model/Tall.nc](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3/hidden-layer-20/seed-default/model). Additional seeds and ANN with more neurons are provided in [CM26_ML_models/ocean3d/subfilter/FGR3/](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/CM26_ML_models/ocean3d/subfilter/FGR3)
* The files required to run online experiments in idealized and global ocean configurations are provided in folders [configurations/Neverworld2](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/NeverWorld2) and [configurations/OM4](https://github.com/m2lines/ANN-momentum-mesoscale/tree/main/configurations/OM4), respectively

# Training ANN on global ocean data CM2.6
The ANN parameterization with local dimensional scaling is executed in Python code in [training-on-CM2.6/helpers/state_functions.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/state_functions.py#L1328). 
* Download training dataset from [Zenodo](https://zenodo.org/records/15328410).
```
wget https://zenodo.org/records/15328410/files/factor-15.zip
wget https://zenodo.org/records/15328410/files/factor-12.zip
wget https://zenodo.org/records/15328410/files/factor-9.zip
wget https://zenodo.org/records/15328410/files/factor-4.zip
```
* Unpack dataset:
```
unzip factor-15.zip
unzip factor-12.zip
unzip factor-9.zip
unzip factor-4.zip
```
* Provide path to the training data in [cm26.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/cm26.py#L16)
* Provide path where to save trained ANNs in [train_script.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py#L38)

## Training algorithm and evaluation
[Training loop](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/train_ann.py#L110) is executed on CPUs via the following script and takes ~9 hours on 4CPU cores and 64GB memory:
```
cd src/training-on-CM2.6/scripts/
python train_script.py
```
This script [contains](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/train_script.py#L19-L32) the default hyperparameters used in the paper. The skill on the testing dataset will be available in `{path_save}/skill-test/factor-{factor}.nc` and log of training/validation losses in `{path_save}/model/logger.nc`.

## How training data was generated
In case you want to recreate training data yourself instead of downloading it from Zenodo, here are the instructions.
### Downloading raw CM2.6 data
* Set where to save raw data in [download_raw_data.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/download_raw_data.py#L11)
* Run the script:
```
cd src/training-on-CM2.6/scripts/
python download_raw_data.py
```

### Filtering and coarsegraining
* Provide path to `rawdata` in [training-on-CM2.6/helpers/cm26.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/cm26.py#L120)
* Provide the path where to save coarsened data in script [generate_3d_datasets.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/scripts/generate_3d_datasets.py#L25)
* Dataset for each coarsegraining factor (`4,9,12,15`) is generated with the script:
```
cd src/training-on-CM2.6/scripts/
python generate_3d_datasets.py --factor=4
```
* Provide the path to newly generated coarse datasets in [cm26.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/training-on-CM2.6/helpers/cm26.py#L16)

## Filtering and offline analysis in idealized configuration NW2
Filtered dataset with diagnosed subfilter fluxes in idealized configuration NW2 is constructed using scripts [filter-NW2-data.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/offline-NW2/filter-NW2-data.py) and [filter-interfaces-GM-filter.py](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/src/offline-NW2/filter-interfaces-GM-filter.py). Second script is optional and used only to more accurately estimate APE in outcropping regions. Offline prediction of subfilter fluxes and evaluation of offline skill is present in notebook [offline-analysis-NW2.ipynb](https://github.com/m2lines/ANN-momentum-mesoscale/blob/main/notebooks/offline-analysis-NW2.ipynb).

# Training ANN on buoyancy fluxes (this fork)

The buoyancy-flux pipeline mirrors the momentum-flux pipeline above: same coarsened CM2.6 datasets, same ANN architecture, with a separate training routine targeting subfilter ρ-fluxes.

* Training routine: [src/training-on-CM2.6/helpers/train_rho_fluxes.py](src/training-on-CM2.6/helpers/train_rho_fluxes.py)
* Training script: [src/training-on-CM2.6/scripts/train_script_rho_fluxes.py](src/training-on-CM2.6/scripts/train_script_rho_fluxes.py)
* Buoyancy-flux diagnostics added to [src/training-on-CM2.6/helpers/cm26.py](src/training-on-CM2.6/helpers/cm26.py)
* Slurm wrappers for dataset generation: `src/training-on-CM2.6/scripts/slurm_generate_datasets_{4,9,12,15}.sh`

Run training:
```
cd src/training-on-CM2.6/scripts/
python train_script_rho_fluxes.py
# or via Slurm:
sbatch slurm_train_ann.sh
```

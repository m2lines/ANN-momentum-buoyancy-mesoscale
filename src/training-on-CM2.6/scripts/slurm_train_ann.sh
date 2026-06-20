#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220GB
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --job-name=ANN_train_rho
#SBATCH --output=train_rho_%x_%j.out
#SBATCH --error=train_rho_%x_%j.err

### Train the rho-flux ANN on regenerated CM2.6 (post Greene->torch migration).
### Reads train+validate from CM26_DATA_ROOT (default /scratch/$USER/CM26_datasets/
### ocean3d); writes the model to /scratch/$USER/mom6/CM26_ML_models/FGR<FGR>/
### <PATH_SAVE>/model/ann_instance.nc. Defaults reproduce the canonical EXP0 config
### (hidden [32,32], 500 iters, stencil 3).
### Usage (CPU, partition cs):
###   sbatch --export=ALL,PATH_SAVE=EXP1,SEED=0 --job-name=train_EXP1 slurm_train_ann.sh
### Usage (GPU, ~18x faster step; pass a GPU partition + gres on the CLI):
###   sbatch --export=ALL,DEVICE=cuda,PATH_SAVE=EXP2,SEED=0 -p l40s_public --gres=gpu:1 \
###          --job-name=train_EXP2 slurm_train_ann.sh "[32,32]"
### HIDDEN is a positional arg (NOT --export: commas in "[32,32]" collide with --export's
### comma separator); STENCIL/SEED/ITERS/DEVICE/PATH_SAVE via --export are fine.
HIDDEN="${1:-[32,32]}"
STENCIL="${STENCIL:-3}"
ITERS="${ITERS:-500}"
SEED="${SEED:-0}"
PATH_SAVE="${PATH_SAVE:-EXP1}"
DEVICE="${DEVICE:-cpu}"

NVFLAG=""
[ "$DEVICE" = "cuda" ] && NVFLAG="--nv"   # expose the GPU to the container only when needed

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec $NVFLAG --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; \
        time python -u train_script_rho_fluxes.py \
            --hidden_layers='$HIDDEN' --stencil_size=$STENCIL --time_iters=$ITERS \
            --seed=$SEED --device=$DEVICE --path_save=$PATH_SAVE"

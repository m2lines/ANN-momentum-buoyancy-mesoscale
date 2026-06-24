#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220GB
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --job-name=ANN_train_sf
#SBATCH --output=train_rho_%x_%j.out
#SBATCH --error=train_rho_%x_%j.err

### Like slurm_train_ann.sh, but FORWARDS --subfilter so training + the auto skill-test can
### point at an alternative dataset (e.g. subfilter-neutral / subfilter-sigma0 from
### generate_density_flux.py). Defaults reproduce the canonical EXP0 config otherwise.
### Usage (GPU):
###   sbatch --export=ALL,DEVICE=cuda,SUBFILTER=subfilter-neutral,PATH_SAVE=EXP_neutral \
###          -p l40s_public --gres=gpu:1 --job-name=train_neutral slurm_train_subfilter.sh "[32,32]" "[4,9,12]"
HIDDEN="${1:-[32,32]}"
FACTORS="${2:-[4,9,12,15]}"
STENCIL="${STENCIL:-3}"
ITERS="${ITERS:-500}"
SEED="${SEED:-0}"
ROTATED="${ROTATED:-0}"
LOSS="${LOSS:-mse}"
SUBFILTER="${SUBFILTER:-subfilter}"            # subfilter | subfilter-neutral | subfilter-sigma0
PATH_SAVE="${PATH_SAVE:-EXP1}"
DEVICE="${DEVICE:-cpu}"

NVFLAG=""
[ "$DEVICE" = "cuda" ] && NVFLAG="--nv"

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec $NVFLAG --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; \
        time python -u train_script_rho_fluxes.py \
            --hidden_layers='$HIDDEN' --factors='$FACTORS' --stencil_size=$STENCIL --time_iters=$ITERS \
            --seed=$SEED --device=$DEVICE --rotated=$ROTATED --loss=$LOSS \
            --subfilter=$SUBFILTER --path_save=$PATH_SAVE"

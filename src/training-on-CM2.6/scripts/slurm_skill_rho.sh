#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=4:00:00
#SBATCH --partition=cs
#SBATCH --job-name=skill_rho
#SBATCH --output=skill_rho_%x_%j.out
#SBATCH --error=skill_rho_%x_%j.err

### Offline skill (R2F/corr_F) of the canonical rho-flux ANN on regenerated data.
### Factors passed as a positional arg (NOT --export: commas in "[4,9,12,15]" collide
### with --export's comma separator). SPLIT/DEVICE via --export is fine.
### Usage (CPU): sbatch --export=ALL,SPLIT=validate --job-name=skill_validate slurm_skill_rho.sh "[4,9,12,15]"
### Usage (GPU, ~10x via batched predict): sbatch --export=ALL,SPLIT=test,DEVICE=cuda \
###   -p l40s_public --gres=gpu:1 --job-name=skill_test_gpu slurm_skill_rho.sh "[4,9,12,15]"
FACTORS="${1:-[9]}"
SPLIT="${SPLIT:-test}"
NTIME="${NTIME:-0}"      # 0 = all snapshots; set e.g. 24 to subsample train to test's count
DEVICE="${DEVICE:-cpu}"  # cuda runs the batched inference on GPU (~10x)

NVFLAG=""
[ "$DEVICE" = "cuda" ] && NVFLAG="--nv"

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec $NVFLAG --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "export FACTORS='$FACTORS' SPLIT='$SPLIT' NTIME='$NTIME' DEVICE='$DEVICE'; source /ext3/env.sh; time python -u compute_skill_rho.py"

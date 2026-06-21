#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180GB
#SBATCH --time=06:00:00
#SBATCH --job-name=cube_cnn
#SBATCH --output=cube_%x_%j.out
#SBATCH --error=cube_%x_%j.err

### Phase A2 cube/CNN ceiling probe (see ../CEILING_STUDY_PLAN.md). GPU job.
### Usage:
###   sbatch --export=ALL,FACTOR=15,EPOCHS=300 -p h200_public --account=torch_pr_347_courant \
###          --gres=gpu:1 --job-name=cube_f15 slurm_train_cube.sh
### Knobs via --export: FACTOR EPOCHS WIDTH DILATIONS CROP BATCH LR WD MAXTRAIN SEED OUT.

FACTORS="${FACTORS:-4:9:12:15}"      # ':'-separated (comma-safe for --export)
EPOCHS="${EPOCHS:-300}"
WIDTH="${WIDTH:-128}"
DILATIONS="${DILATIONS:-1:2:4:8}"
CROP="${CROP:-128}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-3}"
WD="${WD:-1e-5}"
MAXTRAIN="${MAXTRAIN:-24}"
MAXTEST="${MAXTEST:-12}"
SEED="${SEED:-0}"
FIELDS="${FIELDS:-}"                 # empty -> script default (full field set)
OUT="${OUT:-}"                       # empty -> script default path

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --nv --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; \
        FACTORS=$FACTORS EPOCHS=$EPOCHS WIDTH=$WIDTH DILATIONS=$DILATIONS CROP=$CROP \
        BATCH=$BATCH LR=$LR WD=$WD MAXTRAIN=$MAXTRAIN MAXTEST=$MAXTEST SEED=$SEED DEVICE=cuda \
        ${FIELDS:+FIELDS=$FIELDS} ${OUT:+OUT=$OUT} \
        time python -u train_cube_cnn.py"

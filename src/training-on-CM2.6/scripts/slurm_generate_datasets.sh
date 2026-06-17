#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --begin=now
#SBATCH --time=48:00:00
#SBATCH --partition=cs
#SBATCH --job-name=gen_rho
#SBATCH --output=gen_rho_%x_%j.out
#SBATCH --error=gen_rho_%x_%j.err

### Parameterized dataset regeneration (post Greene->torch migration).
### Raw input: /scratch/pp2681/.../rawdata (CM26_RAWDATA); output: /scratch/$USER (CM26_DATA_ROOT default).
### Usage:
###   sbatch --export=ALL,FACTOR=4,MODE=test --job-name=gen_test_4  slurm_generate_datasets.sh   # test split only
###   sbatch --export=ALL,FACTOR=4,MODE=full --job-name=gen_full_4  slurm_generate_datasets.sh   # train+validate+test
FACTOR=${FACTOR:-9}
MODE=${MODE:-test}
if [ "$MODE" = "test" ]; then DSARG="--datasets test";
elif [ "$MODE" = "trainval" ]; then DSARG="--datasets train validate";
else DSARG=""; fi   # full = train validate test (the generate default)

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "export CM26_RAWDATA=/scratch/pp2681/CM26_datasets/ocean3d/rawdata; \
                  source /ext3/env.sh; \
                  time python -u generate_3d_datasets.py --factor=$FACTOR $DSARG"

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --partition=cs
#SBATCH --job-name=neut_chk
#SBATCH --output=neut_chk_%j.out
#SBATCH --error=neut_chk_%j.err
cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts
singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; time python -u eval_neutral_check.py"

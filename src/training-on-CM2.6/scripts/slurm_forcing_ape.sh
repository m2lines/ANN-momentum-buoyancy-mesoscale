#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220GB
#SBATCH --time=2:00:00
#SBATCH --partition=cs
#SBATCH --job-name=force_ape
#SBATCH --output=force_ape_%j.out
#SBATCH --error=force_ape_%j.err

### Score the forcing- and APE-sink metrics (eval_forcing_ape.py) for EXP0 on factor-9 test.
### CPU + big RAM: the skill build now carries rho and forms float64 transport/forcing fields.
cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; time python -u eval_forcing_ape.py"

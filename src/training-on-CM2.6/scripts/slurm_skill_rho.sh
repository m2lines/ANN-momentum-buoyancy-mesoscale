#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --partition=cs
#SBATCH --job-name=skill_rho
#SBATCH --output=skill_rho_%x_%j.out
#SBATCH --error=skill_rho_%x_%j.err

### Offline skill (R2F/corr_F) of the canonical rho-flux ANN on regenerated test data.
### Usage: sbatch --export=ALL,FACTORS="[9]" --job-name=skill_rho_9 slurm_skill_rho.sh
FACTORS=${FACTORS:-[9]}

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "export FACTORS='$FACTORS'; source /ext3/env.sh; time python -u compute_skill_rho.py"

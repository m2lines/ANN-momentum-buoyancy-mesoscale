#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --begin=now
#SBATCH --time=12:00:00
#SBATCH --partition=cs
#SBATCH --job-name=gen_rho_test_9
#SBATCH --output=gen_rho_test_9_%j.out
#SBATCH --error=gen_rho_test_9_%j.err
# CPU job on the torch cluster's general CPU partition `cs` (cpu_short rejects
# with "CPU job setup is not valid" for this account). Default account works.

### Regenerate the TEST split for factor-9 with rho fluxes (Fx,Fy), after the
### Greene -> torch migration that lost the /vast coarse datasets.
###   raw input : /scratch/pp2681/CM26_datasets/ocean3d/rawdata  (Pavel's surviving copy; CM26_RAWDATA)
###   output    : /scratch/$USER/CM26_datasets/ocean3d/subfilter/FGR3/factor-9/  (CM26_DATA_ROOT default)
### Test-only is the proof of concept; full train+validate regen follows if this works.

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "export CM26_RAWDATA=/scratch/pp2681/CM26_datasets/ocean3d/rawdata; \
                  source /ext3/env.sh; \
                  time python -u generate_3d_datasets.py --factor=9 --datasets test"

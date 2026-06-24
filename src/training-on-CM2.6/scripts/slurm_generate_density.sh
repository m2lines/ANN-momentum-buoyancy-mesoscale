#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=cs
#SBATCH --job-name=gen_density
#SBATCH --output=gen_density_%j.out
#SBATCH --error=gen_density_%j.err

# Generate a density-variant coarse dataset (sigma0 or neutral) via the full production pipeline.
# CPU+RAM heavy (full-res flux + momentum SGS). One (DENSITY, SPLIT) per job.
#   sbatch --export=ALL,DENSITY=neutral,SPLIT=train slurm_generate_density.sh
#   sbatch --export=ALL,DENSITY=sigma0,SPLIT=test,NSNAP=1 slurm_generate_density.sh   # smoke

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "source /ext3/env.sh; \
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; \
        time DENSITY=${DENSITY:-neutral} SPLIT=${SPLIT:-test} NSNAP=${NSNAP:-0} FACTOR=${FACTOR:-9} python -u generate_density_flux.py"

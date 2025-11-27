#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=128GB
#SBATCH --begin=now
#SBATCH --time=48:00:00
#SBATCH --job-name=generate_datasets
#SBATCH --output=generate_datasets_%j.out
#SBATCH --error=generate_datasets_%j.err

### Configuration for generating datasets
### Recommended: --cpus-per-task=14 --mem=128GB --time=48:00:00
### Can adjust memory based on dataset size (64GB, 128GB, or 256GB)

# Navigate to script directory
cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

# Generate datasets using Pavel's container
# Default: generates all datasets (train, validate, test) with rho fluxes enabled
singularity exec --nv --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=12"

# Examples of other configurations (uncomment to use):

# Generate only validation dataset
#singularity exec --nv --overlay /scratch/$USER/Pavel_container.ext3:ro \
#    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#    /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=4 --FGR=3 --datasets validate --add_rho_fluxes=1"

# Generate with different factor
#singularity exec --nv --overlay /scratch/$USER/Pavel_container.ext3:ro \
#    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#    /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=9 --FGR=3 --add_rho_fluxes=1"

# Generate without rho fluxes
#singularity exec --nv --overlay /scratch/$USER/Pavel_container.ext3:ro \
#    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#    /bin/bash -c "source /ext3/env.sh; time python -u generate_3d_datasets.py --factor=4 --FGR=3 --add_rho_fluxes=0"

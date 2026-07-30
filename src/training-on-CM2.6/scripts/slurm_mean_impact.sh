#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=8:00:00
#SBATCH --job-name=mean_impact
#SBATCH --output=mean_impact_%x_%j.out
#SBATCH --error=mean_impact_%x_%j.err

### Time-mean eddy-impact maps (forcing G and APE sink a, full column and below the mixed layer),
### diagnosed vs ANN, for eval_mean_impact.py. One factor per job -- the factors are independent and
### factor-4 is ~5x the cost of factor-9, so running them separately keeps the tail off the others.
###
### *** RUN THIS ON CPU. DO NOT REQUEST A GPU. ***
### Tried 2026-07-29 on l40s_public and it drew a low-utilisation warning from HPC at AveUtil 0.07,
### with two of four jobs auto-cancelled at the 2 h mark (threshold is 50%). The reasoning that led
### there was wrong in an instructive way: the ANN forward pass is ~80% of per-snapshot time WHEN
### MEASURED ON CPU, but putting it on a GPU removes it as the bottleneck and leaves the CPU tail --
### the ~30 SGS_skill_rho metrics we discard (13.7 s), the gsw mixed-layer calculation, the netcdf
### reads, the prediction writes -- to dominate. The GPU then idles between short bursts. What
### matters for picking a resource is what REMAINS after the speedup, not what dominates before it.
###
### SAVE_PRED=1 persists Fx_pred/Fy_pred per snapshot (~19 GB for all four factors). USE_PRED=1
### re-reads them instead of re-running inference: 26 s/snapshot against 67 s, measured at factor-9,
### and CPU-only by construction. Run once with SAVE_PRED=1, then everything after is a cheap
### re-reduction. Both are also what makes a cancelled job resumable rather than lost.
###
### Usage (CPU -- the only supported way):
###   sbatch --export=ALL,SAVE_PRED=1 -p cs --time=8:00:00  --job-name=mi_f9 slurm_mean_impact.sh "[9]"
###   sbatch --export=ALL,USE_PRED=1,SAVE_PRED=1 -p cs --time=20:00:00 --job-name=mi_f4 slurm_mean_impact.sh "[4]"
### factor-4 is ~5x factor-9; give it 20 h. factors 12 and 15 are well under 2 h.
FACTORS="${1:-[9]}"
SPLITS="${SPLITS:-train,validate,test}"
DEVICE="${DEVICE:-cpu}"
OUT="${OUT:-/scratch/$USER/mom6/CM26_ML_models/FGR3/EXP_neutral_all4/mean-impact-mld-full}"

NVFLAG=""
if [ "$DEVICE" = "cuda" ]; then
    # Refuse rather than quietly idle a GPU at ~7% and get auto-cancelled two hours in. If a future
    # version genuinely saturates the device (i.e. the CPU-side per-snapshot work has been cut down),
    # delete this guard deliberately -- do not just re-add --gres.
    echo "ERROR: DEVICE=cuda is not supported for this job; the workload is CPU-bound after the" >&2
    echo "       forward pass and idles the GPU (~7% util, auto-cancelled at 2 h). Use -p cs." >&2
    exit 1
fi

cd /home/db194/ANN-momentum-buoyancy-mesoscale/src/training-on-CM2.6/scripts

singularity exec $NVFLAG --overlay /scratch/$USER/Pavel_container.ext3:ro \
    /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -c "export FACTORS='$FACTORS' SPLITS='$SPLITS' DEVICE='$DEVICE' OUT='$OUT' SAVE_PRED='${SAVE_PRED:-0}' USE_PRED='${USE_PRED:-0}'; \
                  source /ext3/env.sh; time python -u eval_mean_impact.py"

"""Regenerate the offline sensitivity sweep: launch one train+skill slurm job per config.

Reads sensitivity_configs.CONFIGS and submits slurm_train_ann.sh for each, on a GPU node,
writing to /scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity/<name>/ (model + test skill).
Idempotent: a config whose 4 test-skill files already exist is skipped, so re-running just
fills in what's missing.

Usage:
    python run_sensitivity.py --dry-run    # print the sbatch commands, submit nothing
    python run_sensitivity.py              # submit all not-yet-done configs
    python run_sensitivity.py --only st5_h32-32_s0   # submit one config by name
"""
import os
import argparse
import subprocess
from sensitivity_configs import CONFIGS

OUT_ROOT = os.path.expandvars('/scratch/$USER/mom6/CM26_ML_models/FGR3/sensitivity')


def is_done(name):
    d = f'{OUT_ROOT}/{name}/skill-test'
    return all(os.path.exists(f'{d}/factor-{f}.nc') for f in (4, 9, 12, 15))


def is_queued(name):
    # already running/pending? (avoid duplicate submissions on re-run)
    out = subprocess.run(['squeue', '-h', '-n', f'sens_{name}', '-o', '%i'],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def sbatch_cmd(c):
    # HIDDEN is positional (commas collide with --export); the rest via --export.
    export = (f"ALL,DEVICE=cuda,STENCIL={c['stencil_size']},SEED={c['seed']},"
              f"ROTATED={c.get('rotated', 0)},PATH_SAVE=sensitivity/{c['name']}")
    return ['sbatch', f'--export={export}', '-p', 'l40s_public', '--gres=gpu:1',
            f"--job-name=sens_{c['name']}", 'slurm_train_ann.sh', c['hidden_layers']]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--only', default=None, help='submit a single config by name')
    args = ap.parse_args()

    names = [args.only] if args.only else list(CONFIGS)
    for name in names:
        c = CONFIGS[name]
        if is_done(name):
            print('skip (done):', name)
            continue
        if is_queued(name):
            print('skip (queued/running):', name)
            continue
        cmd = sbatch_cmd(c)
        print(' '.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

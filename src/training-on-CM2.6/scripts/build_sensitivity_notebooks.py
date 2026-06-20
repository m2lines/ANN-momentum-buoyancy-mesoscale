"""Build the per-axis sensitivity evaluation notebooks (Part-1 style) and render their figures.

Writes readable notebooks to DB_notebooks/evaluate_model_design/{impact_stencil,
impact_model_sizes,impact_random_seed}.ipynb (plain .ipynb JSON, Pavel_Container kernel), and
runs the same cell code headless to drop the figure PDFs alongside. The notebooks are the
reproducible artifact (re-run them in Jupyter with the Pavel_Container kernel); the PDFs are
pre-rendered for convenience. Run in Pavel_Container.
"""
import os
import json
os.environ['MPLBACKEND'] = 'Agg'   # headless rendering for savefig

NB_DIR = '/home/db194/ANN-momentum-buoyancy-mesoscale/DB_notebooks/evaluate_model_design'

SETUP = """import sys
sys.path.append('../../src/training-on-CM2.6/scripts')
import numpy as np
import matplotlib.pyplot as plt
from sensitivity_eval import depth_mean, depth_profile, FACTORS, SPACING
from sensitivity_configs import cfg

SPAC = [SPACING[f] for f in FACTORS]
def nparams(stencil, hidden):
    sizes = [stencil**2 * 5, *hidden, 2]
    return sum(sizes[i]*sizes[i+1] + sizes[i+1] for i in range(len(sizes)-1))"""

STENCIL = [
    ('md', "# Impact of stencil size (lateral non-locality)\n\n"
           "Offline skill of the rho-flux ANN vs coarse-grid resolution for stencils 1x1, 3x3, 5x5 "
           "(all hidden [32,32], seed 0). Part-2 analog of Part 1's stencil result: a large jump "
           "1x1->3x3 and a small one 3x3->5x5."),
    ('code', SETUP),
    ('code', """sten = {1:'1x1', 3:'3x3', 5:'5x5'}
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for s in (1, 3, 5):
    name = cfg(s, '[32,32]', 0)['name']
    r2 = depth_mean(name, 'R2F'); co = depth_mean(name, 'corr_F')
    ax[0].plot(SPAC, [r2[f] for f in FACTORS], '-o', label=sten[s])
    ax[1].plot(SPAC, [co[f] for f in FACTORS], '-o', label=sten[s])
ax[0].set_ylabel('$R^2$ (depth-mean)'); ax[1].set_ylabel('correlation (depth-mean)')
for a in ax:
    a.set_xlabel('coarse-grid spacing [deg]'); a.legend(title='stencil'); a.grid(alpha=0.3)
plt.suptitle('Impact of stencil size'); plt.tight_layout()
plt.savefig('impact_stencil.pdf', bbox_inches='tight'); plt.show()"""),
    ('md', "### $R^2$ vs depth, by stencil (one panel per resolution)"),
    ('code', """fig, ax = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for j, f in enumerate(FACTORS):
    for s in (1, 3, 5):
        z, r = depth_profile(cfg(s, '[32,32]', 0)['name'], f, 'R2F')
        ax[j].plot(r, z, label=sten[s])
    ax[j].invert_yaxis(); ax[j].set_xlim(0, 1); ax[j].axvline(0, color='k', lw=.5)
    ax[j].set_title('%.1f deg' % SPACING[f]); ax[j].set_xlabel('$R^2$'); ax[j].grid(alpha=0.3)
ax[0].set_ylabel('depth [m]'); ax[0].legend(title='stencil')
plt.suptitle('$R^2$ vs depth, by stencil'); plt.tight_layout()
plt.savefig('impact_stencil_depth.pdf', bbox_inches='tight'); plt.show()"""),
]

MODEL = [
    ('md', "# Impact of model size (capacity)\n\n"
           "Skill vs number of trainable parameters (stencil 3x3). Widths [16,16]->[64,64] carry "
           "seed error bars (seeds 0-2); depth variants [32], [32,32,32] shown too. Part 1 found "
           "skill most sensitive to parameter count, with a plateau."),
    ('code', SETUP),
    ('code', """widths = ['[16,16]', '[32,32]', '[48,48]', '[64,64]']
def overall(name):
    dm = depth_mean(name, 'R2F')
    return np.mean([dm[f] for f in FACTORS]) if dm else np.nan

fig, ax = plt.subplots(figsize=(6.5, 4.5))
xp, yp, ye = [], [], []
for w in widths:
    vals = [overall(cfg(3, w, s)['name']) for s in (0, 1, 2)]
    vals = [v for v in vals if np.isfinite(v)]
    xp.append(nparams(3, eval(w))); yp.append(np.mean(vals)); ye.append(np.std(vals))
ax.errorbar(xp, yp, yerr=ye, fmt='-o', capsize=3, label='width [w,w] (+/- seed std)')
for hid_str, lab in [('[32]', '1 layer [32]'), ('[32,32,32]', '3 layers')]:
    ax.plot(nparams(3, eval(hid_str)), overall(cfg(3, hid_str, 0)['name']), 's', label=lab)
ax.set_xscale('log'); ax.set_xlabel('# trainable parameters'); ax.set_ylabel('$R^2$ (mean over resolution)')
ax.set_title('Skill vs model size (stencil 3x3)'); ax.legend(); ax.grid(alpha=0.3)
plt.savefig('impact_model_sizes.pdf', bbox_inches='tight'); plt.show()"""),
    ('md', "### $R^2$ vs resolution, by width"),
    ('code', """fig, ax = plt.subplots(figsize=(6, 4))
for w in widths:
    dm = depth_mean(cfg(3, w, 0)['name'], 'R2F')
    if dm: ax.plot(SPAC, [dm[f] for f in FACTORS], '-o', label=w)
ax.set_xlabel('coarse-grid spacing [deg]'); ax.set_ylabel('$R^2$ (depth-mean)')
ax.legend(title='hidden'); ax.grid(alpha=0.3); ax.set_title('Skill vs resolution, by width')
plt.savefig('impact_model_sizes_resolution.pdf', bbox_inches='tight'); plt.show()"""),
]

SEED = [
    ('md', "# Impact of random seed\n\n"
           "Five trainings of the baseline (stencil 3x3, hidden [32,32]) differing only in seed. "
           "Shows run-to-run noise, so the stencil/capacity differences can be read as real."),
    ('code', SETUP),
    ('code', """fig, ax = plt.subplots(figsize=(6, 4))
allv = []
for s in range(5):
    dm = depth_mean(cfg(3, '[32,32]', s)['name'], 'R2F')
    if dm:
        y = [dm[f] for f in FACTORS]; allv.append(np.mean(y))
        ax.plot(SPAC, y, '-o', alpha=0.7, label='seed %d' % s)
ax.set_xlabel('coarse-grid spacing [deg]'); ax.set_ylabel('$R^2$ (depth-mean)')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Seed spread: mean $R^2$ = %.4f +/- %.4f (std)' % (np.mean(allv), np.std(allv)))
plt.savefig('impact_random_seed.pdf', bbox_inches='tight'); plt.show()"""),
]

ROTATION = [
    ('md', "# Impact of flow-aligned rotation\n\n"
           "Continuous flow-aligned rotation (Part 1): rotate inputs into the local density-gradient "
           "frame, predict along/across, rotate back. Off (baseline) vs on. The rotation is "
           "equivariance-verified (scripts/test_rotation_equivariance.py)."),
    ('code', SETUP + "\nfrom sensitivity_configs import rotation_axis"),
    ('code', """labs = {rotation_axis()[0]: 'no rotation', rotation_axis()[1]: 'flow-aligned'}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, m, t in zip(ax, ['R2F', 'R2F_along', 'R2F_across'], ['combined', 'along', 'across']):
    for n in rotation_axis():
        dm = depth_mean(n, m)
        if dm: a.plot(SPAC, [dm[f] for f in FACTORS], '-o', label=labs[n])
    a.set_xlabel('coarse-grid spacing [deg]'); a.set_ylabel('$R^2$ (%s)' % t)
    a.legend(); a.grid(alpha=0.3)
plt.suptitle('Impact of flow-aligned rotation'); plt.tight_layout()
plt.savefig('impact_rotation.pdf', bbox_inches='tight'); plt.show()"""),
]

LOSS = [
    ('md', "# Impact of training loss style (MSE vs MAE)\n\n"
           "Training loss on the per-snapshot normalized flux: mean-squared (baseline) vs "
           "mean-absolute (Part 1). The offline R^2 metric is independent of the training loss."),
    ('code', SETUP + "\nfrom sensitivity_configs import loss_style_axis"),
    ('code', """labs = {loss_style_axis()[0]: 'MSE', loss_style_axis()[1]: 'MAE'}
fig, ax = plt.subplots(figsize=(6, 4))
for n in loss_style_axis():
    dm = depth_mean(n, 'R2F')
    if dm: ax.plot(SPAC, [dm[f] for f in FACTORS], '-o', label=labs[n])
ax.set_xlabel('coarse-grid spacing [deg]'); ax.set_ylabel('$R^2$ (depth-mean)')
ax.legend(); ax.grid(alpha=0.3); ax.set_title('Impact of training loss style')
plt.savefig('impact_loss_style.pdf', bbox_inches='tight'); plt.show()"""),
]

NOTEBOOKS = {'impact_stencil': STENCIL, 'impact_model_sizes': MODEL, 'impact_random_seed': SEED,
             'impact_rotation': ROTATION, 'impact_loss_style': LOSS}


def cell(t, src):
    lines = src.splitlines(keepends=True)
    if t == 'md':
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def make_nb(cells):
    return {"cells": [cell(t, s) for t, s in cells],
            "metadata": {"kernelspec": {"display_name": "Pavel_Container", "language": "python", "name": "Pavel_Container"}},
            "nbformat": 4, "nbformat_minor": 5}


os.chdir(NB_DIR)   # so sys.path and savefig resolve relative to the notebook dir
for name, cells in NOTEBOOKS.items():
    with open(f'{NB_DIR}/{name}.ipynb', 'w') as fh:
        json.dump(make_nb(cells), fh, indent=1)
    print('built', name + '.ipynb', flush=True)
    ns = {}
    try:
        for t, src in cells:
            if t == 'code':
                exec(src, ns)
        print('  rendered figures OK', flush=True)
    except Exception as e:
        print('  RENDER FAILED:', repr(e)[:200], flush=True)
print('DONE', flush=True)

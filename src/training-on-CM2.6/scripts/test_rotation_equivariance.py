"""Structural test for the continuous flow-aligned rotation in ANN_rho_inference (rotated=True).

The construction is rotation-equivariant by design: rotate the input fields by any angle phi
(density gradient as a vector -> by phi; strain tensor -> by 2*phi; vorticity invariant) and the
predicted flux must rotate by exactly phi. Holds with a RANDOM untrained net (it is structural,
not learned), so this runs in seconds with no training. Run in Pavel_Container.
"""
import sys
sys.path.append('..')
import numpy as np, torch
from helpers.cm26 import read_datasets, DatasetCM26
from helpers.ann_tools import ANN

torch.manual_seed(0)
ann = ANN([45, 32, 32, 2])               # random, untrained -- equivariance is structural
ds = read_datasets(['test'], [15])['test-15']
d = ds.select2d(time=0, zl=10)           # one 2D slice
nanmax = lambda a: float(np.nanmax(np.abs(a)))

with torch.no_grad():
    p0 = d.state.ANN_rho_inference(ann, stencil_size=3, rotated=True)
Fx0, Fy0 = p0['Fx'].numpy(), p0['Fy'].numpy()
scale = nanmax(Fx0) + nanmax(Fy0)

# sanity: rotated model differs from non-rotated (rotation is actually applied)
with torch.no_grad():
    pn = d.state.ANN_rho_inference(ann, stencil_size=3, rotated=False)
print('rotated vs non-rotated differ by: %.2e (should be O(scale)=%.2e)' % (
    nanmax(p0['Fx'].numpy() - pn['Fx'].numpy()), scale))

def rotate_inputs(d, phi):
    cp, sp = np.cos(phi), np.sin(phi); c2, s2 = np.cos(2*phi), np.sin(2*phi)
    rx, ry = d.data.rhox.values, d.data.rhoy.values
    sxx, sxy = d.data.sh_xx.values, d.data.sh_xy_h.values
    dd = d.data.copy()
    dd['rhox'] = (d.data.rhox.dims, rx*cp - ry*sp)         # vector by phi
    dd['rhoy'] = (d.data.rhoy.dims, rx*sp + ry*cp)
    dd['sh_xx'] = (d.data.sh_xx.dims, sxx*c2 - sxy*s2)     # strain tensor by 2*phi
    dd['sh_xy_h'] = (d.data.sh_xy_h.dims, sxx*s2 + sxy*c2)
    return DatasetCM26(dd, d.param)

print('\n=== equivariance: rotate inputs by phi -> output must rotate by phi ===')
for phi in [0.0, np.pi/4, np.pi/2, 1.3, np.pi, 2.7]:
    cp, sp = np.cos(phi), np.sin(phi)
    with torch.no_grad():
        p1 = rotate_inputs(d, phi).state.ANN_rho_inference(ann, stencil_size=3, rotated=True)
    Fx1, Fy1 = p1['Fx'].numpy(), p1['Fy'].numpy()
    Fx_exp = Fx0*cp - Fy0*sp; Fy_exp = Fx0*sp + Fy0*cp
    err = max(nanmax(Fx1 - Fx_exp), nanmax(Fy1 - Fy_exp))
    print('  phi=%5.2f : max abs err = %.2e  (rel %.2e)  %s' % (
        phi, err, err/scale, 'OK' if err/scale < 1e-5 else 'FAIL'))
print('DONE')

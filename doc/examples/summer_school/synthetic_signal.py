import numpy as np
from dipy.sims.voxel import (single_tensor, single_tensor_odf, 
                             multi_tensor, multi_tensor_odf)
from dipy.data import get_sphere
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk

bvalue = 1000
S0 = 1
SNR = 100

sphere = get_sphere('symmetric362')
bvecs = np.concatenate(([[0, 0, 0]], sphere.vertices))
bvals = np.zeros(len(bvecs)) + bvalue
bvals[0] = 0
gtab = gradient_table(bvals, bvecs)

evals = np.array(([0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]))
evecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
         np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]


S = single_tensor(gtab, evals=evals[1], evecs=evecs[1], snr=SNR) 
odf = single_tensor_odf(sphere.vertices, evals[1], evecs[1])

ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(np.vstack((S[1:], odf)), sphere))
fvtk.show(ren)
fvtk.clear(ren)


S_x, sticks = multi_tensor(gtab, evals, S0, angles=[(0, 0), (90, 0)],
                            fractions=[50, 50], snr=SNR)
odf_x = multi_tensor_odf(sphere.vertices, [0.5, 0.5], evals, evecs)

ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(np.vstack((S_x[1:], odf_x)), sphere))
fvtk.show(ren)





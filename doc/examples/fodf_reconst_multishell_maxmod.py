"""
====================================================
fODF reconstruction from multi-shell measurements
====================================================

First import the necessary modules:
"""
#from dipy.data import three_shells_voxels, two_shells_voxels, get_sphere
from dipy.reconst.shm import sh_to_sf
from dipy.reconst.odf import peaks_from_model, peak_directions
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.viz import fvtk
from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell, get_sphere
from dipy.core.gradients import gradient_table, gradient_table_from_bvals_bvecs
from dipy.core.sphere_stats import angular_similarity

import numpy as np
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)

def maxmod(fodfs) :
    fodf = np.zeros(fodfs.shape[3])

    for i in range(fodfs.shape[3]) :
        if np.abs(fodfs[0,:,:,i]) > np.abs(fodfs[1,:,:,i]) :
            max_tmp = fodfs[1,:,:,i]
        else :
            max_tmp = fodfs[0,:,:,i]

        if np.abs(fodfs[2,:,:,i]) > np.abs(max_tmp) :
            fodf[i] = fodfs[2,:,:,i] 
        else :
            fodf[i] = max_tmp

    return fodf


fetch_sherbrooke_3shell()
img, gtab=read_sherbrooke_3shell()
S0 = 1
angle = 45
SNR = 100
debug = 0
mevals = np.array(([0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]))

evals = np.array([0.0015, 0.0003, 0.0003])
response = (evals, 1)
sphere = get_sphere('symmetric724')
b_vals = gtab.bvals
b_vecs = gtab.bvecs

indices_all = range(len(gtab.bvals))

gtab = gradient_table_from_bvals_bvecs(b_vals[indices_all], b_vecs[indices_all])
S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                         fractions=[50, 50], snr=SNR)
mevecs = [all_tensor_evecs(sticks[0]).T,
          all_tensor_evecs(sticks[1]).T]
odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)


indices_1000 = np.append(np.where(b_vals == 0), np.where(b_vals == 1000))
indices_2000 = np.append(np.where(b_vals == 0), np.where(b_vals == 2000))
indices_3500 = np.append(np.where(b_vals == 0), np.where(b_vals == 3500))


gtab = gradient_table_from_bvals_bvecs(b_vals[indices_1000], b_vecs[indices_1000])

S1, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                         fractions=[50, 50], snr=SNR)
data = np.zeros((3, 1, 1, S1.shape[0]))
data[0,:,:] = S1[:]

gtab = gradient_table_from_bvals_bvecs(b_vals[indices_2000], b_vecs[indices_2000])
S2, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                         fractions=[50, 50], snr=SNR)
gtab = gradient_table_from_bvals_bvecs(b_vals[indices_3500], b_vecs[indices_3500])
S3, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                         fractions=[50, 50], snr=SNR)
data[1,:,:] = S2[:]
data[2,:,:] = S3[:]

csd_model = ConstrainedSphericalDeconvModel(gtab, response)
peaks = peaks_from_model(model=csd_model,
                         data=data,
                         sphere=sphere,
                         relative_peak_threshold=0.25,
                         min_separation_angle=25,
                         return_odf=True,
                         return_sh=True,
                         normalize_peaks=False,
                         npeaks=5,
                         parallel=True,
                         nbr_process=8)

fodfs = peaks.odf
fodfs_sh = peaks.shm_coeff

#print(fodfs_sh.shape)
fodf_sh = maxmod(fodfs_sh)
#print(fodf_sh)

sphere2 = get_sphere('symmetric724').subdivide()
fodf2 = sh_to_sf(fodf_sh, sphere2, 8)
fodf = sh_to_sf(fodf_sh, sphere, 8)
directions, _, _ = peak_directions(fodf2, sphere2, min_separation_angle=15)
directions_gt, _, _ = peak_directions(odf_gt, sphere, min_separation_angle=15)
ang_sim = angular_similarity(directions, directions_gt)
print(angular_similarity(directions[0,:], directions[1,:]) * 180 / np.pi)
print(angular_similarity(directions_gt[0,:], directions_gt[1,:]) * 180 / np.pi)

print len(directions), "direction in the fodf and ", len(directions_gt), " in ground truth and angle similarity is ", ang_sim

r = fvtk.ren()
fvtk.add( r, fvtk.sphere_funcs( np.vstack((odf_gt, fodf, odf_gt)), 
                                sphere ))

fvtk.show(r)

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

import numpy as np
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
"""
"""

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
indices_1000 = np.append(np.where(b_vals == 0), np.where(b_vals == 1000))
indices_2000 = np.append(np.where(b_vals == 0), np.where(b_vals == 2000))
indices_3500 = np.append(np.where(b_vals == 0), np.where(b_vals == 3500))
indices_2000_only = np.where(b_vals == 2000)
indices_3500_only = np.where(b_vals == 3500)

indices1000_2000 = np.append(indices_1000, indices_2000_only)		     	
indices1000_3500 = np.append(indices_1000, indices_3500_only)		     	
indices2000_3500 = np.append(indices_2000, indices_3500_only)		     	

options = ['3-shells', '2-shells_1000-2000', '2-shells_1000-3500', '2-shells_2000-3500',
           '1-shell_1000', '1-shell_2000', '1-shell_3500']
for o in options:
    print(o)

    indices = indices_all
    if o == '2-shells_1000-2000':
        indices = indices1000_2000
    elif o == '2-shells_1000-3500':
        indices = indices1000_3500
    elif o == '2-shells_2000-3500':
        indices = indices2000_3500
    elif o == '1-shell_1000' :
        indices = indices_1000
    elif o == '1-shell_2000' :
        indices = indices_2000
    elif o == '1-shell_3500' :
        indices = indices_3500
    else :
        indices = indices_all

    if debug :
        print(b_vals[indices])

    gtab = gradient_table_from_bvals_bvecs(b_vals[indices], b_vecs[indices])

    # Synthetic signal and ODF
    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                             fractions=[50, 50], snr=SNR)
    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]
    
    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)
    
    csd_model = ConstrainedSphericalDeconvModel(gtab, response)
    peaks = peaks_from_model(model=csd_model,
                             data=S,
                             sphere=sphere,
                             relative_peak_threshold=0.25,
                             min_separation_angle=25,
                             return_odf=True,
                             return_sh=False,
                             normalize_peaks=False,
                             npeaks=5,
                             parallel=False,
                             nbr_process=8)

    fodf = peaks.odf
    directions, _, _ = peak_directions(fodf, sphere, min_separation_angle=25)
    directions_gt, _, _ = peak_directions(odf_gt, sphere, min_separation_angle=25)

    print len(directions), "direction in the fodf and ", len(directions_gt), " in ground truth"

    r = fvtk.ren()
    fvtk.add( r, fvtk.sphere_funcs( np.vstack((odf_gt, fodf)), 
                                    sphere ))
    
    fvtk.show(r)



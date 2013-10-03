"""
===================================
ODF reconstruction and manipulation
===================================

We show how to apply a Q-ball ODF model from [Descoteaux2007]_ and,   
more recent Constant Solid Angle ODF (Q-Ball) model 
from [Aganj2009]_ and  Orientation Probability Density Transform (Opdf)
model from [TristanVega2010]_ to your datasets. We also show how to
visualize the ODFs, compute the genarilized fractional 
anisotropy (GFA) [Tuch2004]_ and  play with spherical harmonics (SH) coefficients.

First import the necessary modules:
"""

import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, OpdtModel, sh_to_sf, normalize_data
from dipy.reconst.odf import peaks_from_model, minmax_normalize

"""
Download and read the data for this tutorial.
"""

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()

"""
img contains a nibabel Nifti1Image object (data) and gtab contains a GradientTable
object (gradient information e.g. b-values). For example to read the b-values
it is possible to write print(gtab.bvals).

Load the raw diffusion data and the affine.
"""

data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

"""
data.shape ``(81, 106, 76, 160)``

Remove most of the background using dipy's mask module.
"""

from dipy.segment.mask import median_otsu

maskdata, mask = median_otsu(data, 3, 2, True, range(0,10))

"""
We instantiate our CSA model with spherical harmonic order of 4
"""

order = 4
csamodel = CsaOdfModel(gtab, order)

"""
`Peaks_from_model` is used to calculate properties of the ODFs (Orientation
Distribution Function) and return for
example the peaks and their indices, or GFA which is similar to FA but for ODF
based models. This function mainly needs a reconstruction model, the data and a
sphere as input. The sphere is an object that represents the spherical discrete
grid where the ODF values will be evaluated.
"""

sphere = get_sphere('symmetric724')

print 'Computing the CSA ODFs...'
csapeaks = peaks_from_model(model=csamodel,
                            data=maskdata,
                            sphere=sphere,
                            relative_peak_threshold=.8,
                            min_separation_angle=45,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True)

GFA = csapeaks.gfa
nib.save(nib.Nifti1Image(GFA.astype(np.float32), img.get_affine()), 'GFA_csa.nii.gz')

"""
GFA.shape ``(81, 106, 76)``

Apart from GFA, csapeaks also has the attributes peak_values, peak_indices and
ODF. peak_values shows the maxima values of the ODF and peak_indices gives us
their position on the discrete sphere that was used to do the reconstruction of
the ODF. In order to obtain the full ODF, return_odf should be True. Before
enabling this option, make sure that you have enough memory.

<<<<<<< HEAD
Finally lets try to visualize the orientation distribution functions for a small
region of interest (ROI) part of the splenium of the corpus callosum (CC).
"""

data_small  = data[20:50,55:85, 38:39] 
=======
Let's visualize the ODFs of a small rectangular area in an axial slice of the
splenium of the corpus callosum (CC).
"""

data_small = maskdata[13:43, 44:74, 28:29]

>>>>>>> 043eb360d57ad18ce5193a9c897b7441f2b161d4
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
csa_odfs = csamodel.fit(data_small).odf(sphere)

from dipy.viz import fvtk
r = fvtk.ren()
<<<<<<< HEAD
fvtk.add(r, fvtk.sphere_funcs(csa_odfs, sphere, colormap='jet'))
#fvtk.show(r)

=======

csaodfs = csamodel.fit(data_small).odf(sphere)

"""
It is common with CSA ODFs to produce negative values, we can remove those using ``np.clip``
"""

csaodfs = np.clip(csaodfs, 0, np.max(csaodfs, -1)[..., None])

fvtk.add(r, fvtk.sphere_funcs(csaodfs, sphere, colormap='jet'))
>>>>>>> 043eb360d57ad18ce5193a9c897b7441f2b161d4
print('Saving illustration as csa_odfs.png')
fvtk.record(r, n_frames=1, out_path='csa_odfs.png', size=(600, 600))

"""
.. figure:: csa_odfs.png
   :align: center

   **Constant Solid Angle ODFs**.

Now, lets see compare with other ODF models available in dipy. We now compute the
the analytical q-ball model of [Descoteaux2007]_ and save its SH coefficients and
GFA. Note that, here, the GFA is computed directly from the SH coefficients.
"""

qballmodel = QballModel(gtab, order, smooth=0.006)
print 'Computing the QBALL ODFs...'
qballfit  = qballmodel.fit(data) 
GFA = qballfit.gfa
SH_coeff   = qballfit._shm_coef
nib.save(nib.Nifti1Image(GFA.astype('float32'), img.get_affine()), 'gfa_qball.nii.gz')    
nib.save(nib.Nifti1Image(SH_coeff.astype('float32'), img.get_affine()), 'qball_odf_sh.nii.gz')

"""
Note that if we want to visualize the q-ball ODFs, we do not need to recompute the qballmodel
in the ROI as we already have the SH coefficients. We can simply get the SH coefficients in
the ROI and project them to the sphere to the obtain the desired ODFs.
"""

odfs = sh_to_sf(SH_coeff[20:50,55:85, 38:39], sphere, order)
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
#fvtk.show(r)

print('Saving illustration as qball_odfs.png')
fvtk.record(r, n_frames=1, out_path='qball_odfs.png', size=(600, 600))
fvtk.clear(r)

"""
.. figure:: qball_odfs.png
   :align: center

   **Q-ball ODFs**.

We see that original q-ball ODFs are not normalized as the CSA ODFs. Hence, typically, for
visualization purposes, we do a min-max normalization before visualization.
"""

print 'Min-max normalizing and visualizing...'
fvtk.add(r, fvtk.sphere_funcs(minmax_normalize(odfs), sphere, colormap='jet'))
#fvtk.show(r)

print('Saving illustration as qball_minmax_odfs.png')
fvtk.record(r, n_frames=1, out_path='qball_minmax_odfs.png', size=(600, 600))
fvtk.clear(r)


"""
.. figure:: qball_minmax_odfs.png
   :align: center

   **Min-max normazlied Q-ball ODFs**.

Next, we compute the Opdf ODF model, save its SH coefficients and GFA, and visualize
its ODFs. 
"""

opdtmodel = OpdtModel(gtab, order, smooth=0.006)
print 'Computing the Opdt ODFs...'
opdtfit = opdtmodel.fit(data) 
SH_coeff   = opdtfit._shm_coef
GFA     = opdtfit.gfa
nib.save(nib.Nifti1Image(GFA.astype('float32'), img.get_affine()), 'gfa_opdt.nii.gz')    
nib.save(nib.Nifti1Image(SH_coeff.astype('float32'), img.get_affine()), 'opdt_odf_sh.nii.gz')

odfs = sh_to_sf(SH_coeff[20:50,55:85, 38:39], sphere, order)
fvtk.add(r, fvtk.sphere_funcs(odfs, sphere, colormap='jet'))
#fvtk.show(r)

print('Saving illustration as opdf_odfs.png')
fvtk.record(r, n_frames=1, out_path='opdt_odfs.png', size=(600, 600))
fvtk.clear(r)

"""
.. figure:: opdt_odfs.png
   :align: center

   **Orientation Probability Density Transform ODFs**.
   
Note that all the save NIFTI GFAs and SH coefficients datasets can be visualized and 
explored interactively using the fibernavigator (https://github.com/scilus/fibernavigator).



.. [Descoteaux2007] Descoteaux, M., et. al. 2007. Regularized, fast, and robust
                    analytical Q-ball imaging. 

.. [Aganj2009] Aganj, I., et. al. 2009. ODF Reconstruction in Q-Ball Imaging With
               Solid Angle Consideration.

.. [TristanVega2010] Tristan-Vega, A., et. al. 2010. A new methodology for estimation of
                     fiber populations in white matter of the brain with Funk-Radon
                     transform.

.. [Tuch2004] Tuch, D. 2004. Q-ball imaging.

.. include:: ../links_names.inc

"""

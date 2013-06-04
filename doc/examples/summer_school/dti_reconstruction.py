import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

# For this example, lets choose a small ROI of the corpus callosum (CC)
mask = np.zeros(data[..., 0].shape)
mask[20:50,55:85,38:39] = 1

import dipy.reconst.dti as dti
tenmodel = dti.TensorModel(gtab)

print('Performing and saving the DTI fit & metric computation in a ROI of the CC...')
tenfit = tenmodel.fit(data, mask)

from dipy.reconst.dti import fractional_anisotropy
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0

fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
nib.save(fa_img, 'ROI_fa.nii.gz')

# similarly for the RGB map and ADC, also called mean_diffuvisty
from dipy.reconst.dti import color_fa, mean_diffusivity, lower_triangular
ADC = mean_diffusivity(tenfit.evals)
nib.save(nib.Nifti1Image(ADC.astype(np.float32), img.get_affine()), 'ROI_adc.nii.gz')

RGB = color_fa(FA, tenfit.evecs)
nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.get_affine()), 'ROI_rgb.nii.gz')

# Now, lets get the tensor coefficients D and format them for them 
# as D = [dxx, dxy, dxz, dyy, dyz, dzz]
tensor_vals = lower_triangular(tenfit.quadratic_form)
correct_order = [0,1,3,2,4,5]
tensor_vals_reordered = tensor_vals[:,:,:,correct_order]
nib.save(nib.Nifti1Image(tensor_vals_reordered.astype(np.float32), img.get_affine()), 
         'ROI_coefs.nii.gz')

# compute a mask from FSL
# bet HARDI150.nii.gz mask -m 
mask_file = 'mask_mask.nii.gz'
mask_img = nib.load(mask_file)
mask = mask_img.get_data()

print('Performing DTI full brain ... (be patient)')
tenfit = tenmodel.fit(data, mask)
FA = fractional_anisotropy(tenfit.evals)
FA[np.isnan(FA)] = 0
fa_img = nib.Nifti1Image(FA.astype(np.float32), img.get_affine())
nib.save(fa_img, 'fa.nii.gz')
ADC = mean_diffusivity(tenfit.evals)
nib.save(nib.Nifti1Image(ADC.astype(np.float32), img.get_affine()), 'adc.nii.gz')
RGB = color_fa(FA, tenfit.evecs)
nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), img.get_affine()), 'rgb.nii.gz')
tensor_vals = lower_triangular(tenfit.quadratic_form)
correct_order = [0,1,3,2,4,5]
tensor_vals_reordered = tensor_vals[:,:,:,correct_order]
nib.save(nib.Nifti1Image(tensor_vals_reordered.astype(np.float32), img.get_affine()), 
         'tensors_coeffs.nii.gz')



print('Computing tensor ODFs in a part of the splenium of the CC')
data_small  = data[20:50,55:85, 38:39]
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(tenmodel.fit(data_small).odf(sphere),
							  sphere, colormap=None))
fvtk.show(r)
print('Saving illustration as tensor_odfs.png')
fvtk.record(r, n_frames=1, out_path='tensor_odfs.png', size=(600, 600))




# TO DO: visualize peaks

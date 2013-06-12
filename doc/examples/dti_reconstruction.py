import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.core.gradients import gradient_table

# compute a mask from FSL
# bet HARDI150.nii.gz mask -m 
mask_file = 'mask_mask.nii.gz'
mask_img = nib.load(mask_file)
mask = mask_img.get_data()

data_file = 'DTI_from_dcm2nii.nii.gz'
img = nib.load(data_file)
data = img.get_data()

bvals = np.loadtxt('bval')
bvecs = np.loadtxt('bvec')
gtab = gradient_table(bvals, bvecs)

import dipy.reconst.dti as dti
tenmodel = dti.TensorModel(gtab)

from dipy.reconst.dti import fractional_anisotropy, color_fa, mean_diffusivity, lower_triangular
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


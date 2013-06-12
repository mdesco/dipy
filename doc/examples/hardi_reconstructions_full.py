import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, sh_to_sf
from dipy.reconst.odf import gfa

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

mask_file = 'mask_mask.nii.gz'
mask_img = nib.load(mask_file)
mask = mask_img.get_data()

sphere2 = get_sphere('symmetric362')

qmodel = QballModel(gtab, 4, smooth=0.006)
print 'Computing the ODFs and GFA on the full brain (be patient)...'
qfit = qmodel.fit(data, mask) 
SH_coeff   = qfit._shm_coef
GFA     = gfa(qfit.odf(sphere2))
nib.save(nib.Nifti1Image(GFA.astype('float32'), img.get_affine()), 'gfa.nii.gz')    
nib.save(nib.Nifti1Image(SH_coeff.astype('float32'), img.get_affine()), 'qball_odf_sh.nii.gz')

<<<<<<< HEAD
import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import dwi_bet_filter
from scipy.ndimage import generate_binary_structure, binary_dilation
from dipy.segment.mask import hist_mask

files = ['./test_datasets_multicite_all_companies/Stanford/b0.nii.gz', 
         './test_datasets_multicite_all_companies/3T/GE/b0.nii.gz', 
         './test_datasets_multicite_all_companies/3T/Siemens/b0.nii.gz', 
         './test_datasets_multicite_all_companies/3T/Philips/b0.nii.gz',
         './test_datasets_multicite_all_companies/1.5T/GE/b0.nii.gz', 
         './test_datasets_multicite_all_companies/1.5T/Siemens/b0.nii.gz']

for f in files :
    print(f)
    img = nib.load(f)
    data = img.get_data()

    b0 = data[:,:,:, 0]
        
    dwi_mask, mask = dwi_bet_filter(data, 4, 4, autocrop=False)
    mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
    print(f+'_mask.nii.gz')
    nib.save(mask_img, f+'_mask.nii.gz')

    epi_mask = hist_mask(b0)
    epi_mask_img = nib.Nifti1Image(epi_mask.astype(np.float32), img.get_affine())
    print(f+'_epi_mask.nii.gz')
    nib.save(epi_mask_img, f+'_epi_mask.nii.gz')
    
=======
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.segment.mask import hist_mask

#img, gtab = read_stanford_hardi()


files = ['./test_bench/Stanford/b0.nii.gz', 
         './test_bench/3T/GE/b0.nii.gz', 
         './test_bench/3T/Siemens/b0.nii.gz', 
         './test_bench/3T/Philips/b0.nii.gz',
         './test_bench/1.5T/GE/b0.nii.gz', 
         './test_bench/1.5T/Siemens/b0.nii.gz']

for f in files:

    img = nib.load(f)

    b0 = img.get_data().squeeze()
    affine = img.get_affine()

    #b0 = np.mean(data[..., 0:9], axis=-1)

    mask = hist_mask(b0)

    #nib.save(nib.Nifti1Image(b0, affine), f + 'b0.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype('byte'), affine), f + '_mask.nii.gz')



>>>>>>> 12e8076443e6764af035f6e06948f9140e356f74

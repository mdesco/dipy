from nibabel.nicom.dicomreaders import read_mosaic_dir

#dirname='/Volumes/BrainData/RawData/UNF_Montreal/3T_Siemens/JF-Gagnon/03-DTI'
#dirname='/Volumes/BrainData/RawData/NYMU_Taiwan/Spatial_2x2x2_mm3/Sujet01/MultipleShellSession/one_shell_b1000/dicom'
dirname='/Volumes/BrainData/RawData/CHUS_Fleurimont/Fortin/Harvey_5_juin_2013/13060416/38070000'
data, affine, bvals, bvecs = read_mosaic_dir(dirname,globber='*',check_is_dwi=True)




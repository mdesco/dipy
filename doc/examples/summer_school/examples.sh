python raw_data.py

anatomist ~/.dipy/stanford_hardi/HARDI150.nii.gz & 

bet ~/.dipy/stanford_hardi/HARDI150.nii.gz mask -m 

python dti_reconstruction.py

python hardi_reconstructions.py

python deterministic_tracking.py




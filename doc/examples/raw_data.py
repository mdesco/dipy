import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi

fetch_stanford_hardi()
# the data is in your $HOME/.dipy directory
# launch your favorite viewer and look at the data. For example, anatomist.

# Loading the data into python
img, gtab = read_stanford_hardi()
data = img.get_data()
print('data.shape (%d, %d, %d, %d)' % data.shape)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax  = fig.add_subplot(111)
# slice_b0 = data[:,:,38,0]
# plt.imshow(slice_b0.T)

# slice_dwi100 = data[:,:,38,100]
# plt.imshow(slice_dwi100.T)


# Visualizing b-vectors
#bvec_name = '/Users/mdescoteaux/.dipy/stanford_hardi/HARDI150.bvec'
import os
bvec_name = os.environ['HOME'] + '/.dipy/stanford_hardi/HARDI150.bvec'
bvecs = np.loadtxt(bvec_name).T

from dipy.viz import fvtk
ren = fvtk.ren()
sphere_actor = fvtk.point(bvecs, colors=fvtk.red, opacity=1, point_radius=0.05, theta=10, phi=20)
fvtk.add(ren, sphere_actor)
fvtk.show(ren)
#print('Saving illustration b-vecs.png')
#fvtk.record(ren, n_frames=1, out_path='bvecs.png', size=(600, 600))






import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, sh_to_sf
from dipy.data import get_sphere
from dipy.viz import fvtk

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
sphere = get_sphere('symmetric724')

data_small  = data[20:50,55:85, 38:39]


order = 4
qballmodel = QballModel(gtab, order, smooth=0.006)
print 'Computing the CSA ODFs...'
qballfit = qballmodel.fit(data_small)
qballodfs = qballfit.odf(sphere)


ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(qballodfs, sphere, colormap='jet'))
fvtk.show(ren)
print('Saving illustration as qballodfs.png')
fvtk.record(ren, n_frames=1, out_path='qballodfs.png', size=(600, 600))
fvtk.clear(r)


# min-max normalize ODFs
print 'Min-max normalizing and visualizing...'
minmax_odfs = odfs
fvtk.add(r, fvtk.sphere_funcs(minmax_odfs, sphere, colormap='jet'))
fvtk.show(r)

print('Saving illustration as qball_minmax_odfs.png')
fvtk.record(r, n_frames=1, out_path='qball_minmax_odfs.png', size=(600, 600))
fvtk.clear(r)






csamodel = CsaOdfModel(gtab, order, smooth=0.006)
print 'Computing the CSA ODFs...'
csafit = csamodel.fit(data_small)
csaodfs = csafit.odf(sphere)

ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(csaodfs, sphere, colormap='jet'))
fvtk.show(ren)
print('Saving illustration as csa_odfs.png')
fvtk.record(ren, n_frames=1, out_path='csa_odfs.png', size=(600, 600))


# TO DO: visualize peaks

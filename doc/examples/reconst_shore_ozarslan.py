
from dipy.reconst.shore_ozarslan import ShoreOzarslanModel
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
from dipy.data import fetch_isbi2013_2shell, read_isbi2013_2shell, get_sphere
from dipy.core.gradients import gradient_table


fetch_isbi2013_2shell()
img, gtab = read_isbi2013_2shell()
data = img.get_data()
data_small = data[10:40, 22, 10:40]

print(data_small.shape)

radial_order = 8
asm = ShoreOzarslanModel(gtab, radial_order=radial_order,
                         laplacian_regularization=True,
                         laplacian_weighting=0.2)
                         
                         
asmfit = asm.fit(data_small)
sphere = get_sphere('symmetric724')
odf = asmfit.odf(sphere)
print('odf.shape (%d, %d, %d)' % odf.shape)


r = fvtk.ren()
sfu = fvtk.sphere_funcs(odf[:, None, :], sphere, colormap='jet')
sfu.RotateX(-90)
fvtk.add(r, sfu)
fvtk.show(r) #, n_frames=1, out_path='odfs.png', size=(600, 600))





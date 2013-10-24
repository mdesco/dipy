import warnings
import numpy as np
from numpy.testing import (assert_equal, run_module_suite)
from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   odf_sh_to_sharp)
from dipy.reconst.odf import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.shm import sf_to_sh, sh_to_sf, QballModel, CsaOdfModel


def test_csdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    csd = ConstrainedSphericalDeconvModel(gtab, response)
    
    csd_fit = csd.fit(S)

    assert_equal(csd_fit.shm_coeff[0] > 0, True)

    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSphericalDeconvModel(gtab, response, sh_order=10)
        assert_equal(len(w) > 0, True)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
        assert_equal(len(w) > 0, False)


def test_odfdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    e1 = 15.0
    e2 = 3.0
    ratio = e2 / e1

    csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)
    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)


    assert_equal(ang_sim > 1.9, True)

    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSDTModel(gtab, ratio, sh_order=10)
        assert_equal(len(w) > 0, True)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSDTModel(gtab, ratio, sh_order=8)
        assert_equal(len(w) > 0, False)



def test_r2_term_odf_sh_to_sharp():

    SNR = None
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')  #get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    sphere = get_sphere('symmetric724')

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)
    
    
    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)


    csa = CsaOdfModel(gtab, sh_order=8, assume_normed=True)
    
    csafit = csa.fit(S)
    odf_csa = csafit.odf(sphere)


    odfs_sh3 = sf_to_sh(odf_csa, sphere, sh_order=8, basis_type=None)
    fodf_sh3 = odf_sh_to_sharp(odfs_sh3, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=0.1, r2_term=True)
    

    odfs_sh = sf_to_sh(odf_gt, sphere, sh_order=8, basis_type=None)
    fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=0.1, r2_term=True)

    fodf = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None)

    fodf_sh2 = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=0.1, r2_term=False)

    fodf2 = sh_to_sf(fodf_sh2, sphere, sh_order=8, basis_type=None)
    fodf3 = sh_to_sf(fodf_sh3, sphere, sh_order=8, basis_type=None)

    from dipy.viz import fvtk
    r = fvtk.ren()
    fvtk.add( r, fvtk.sphere_funcs( np.vstack((fodf, fodf2, odf_gt, fodf3)), sphere ))
    fvtk.show( r )
    
    directions_gt, _, _ = peak_directions(odf_gt, sphere)
    directions, _, _ = peak_directions(fodf, sphere)
    directions2, _, _ = peak_directions(fodf2, sphere)
    directions3, _, _ = peak_directions(fodf3, sphere)
    
    ang_sim = angular_similarity(directions_gt, directions)
    assert_equal(ang_sim > 1.9, True)
    ang_sim = angular_similarity(directions_gt, directions3)
    assert_equal(ang_sim > 1.9, True)
    ang_sim = angular_similarity(directions_gt, directions2)
    assert_equal(ang_sim > 1.9, True)



def test_odf_sh_to_sharp():

    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(10, 0), (100, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')

    qb = QballModel(gtab, sh_order=8, assume_normed=True)

    qbfit = qb.fit(S)
    odf_gt = qbfit.odf(sphere)

    Z = np.linalg.norm(odf_gt)

    odfs_gt = np.zeros((3, 1, 1, odf_gt.shape[0]))
    odfs_gt[:,:,:] = odf_gt[:]

    odfs_sh = sf_to_sh(odfs_gt, sphere, sh_order=8, basis_type=None)

    odfs_sh /= Z

    fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=1.)

    fodf = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None)

    directions2, _, _ = peak_directions(fodf[0, 0, 0], sphere)

    assert_equal(directions2.shape[0], 2)


if __name__ == '__main__':
    #run_module_suite()
    test_r2_term_odf_sh_to_sharp()

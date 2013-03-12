import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.reconst.odf import (OdfFit, OdfModel, gfa, peaks_from_model, peak_directions,
                              peak_directions_nl)
from dipy.core.subdivide_octahedron import create_unit_hemisphere
from dipy.core.sphere import unit_icosahedron, unit_octahedron, Sphere
from nose.tools import (assert_almost_equal, assert_equal, assert_raises,
                        assert_true)
from dipy.reconst.shm import (sph_harm_ind_list, sph_harm_lookup, smooth_pinv,
                              real_sph_harm, sh_to_sf, sf_to_sh)
from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (single_tensor, single_tensor_odf,
                            multi_tensor, multi_tensor_odf)
from dipy.core.gradients import gradient_table
from scipy.special import lpn
from dipy.reconst.shm import (sph_harm_ind_list, 
                              real_sph_harm, 
                              lazy_index,
                              sh_to_sf)
from dipy.core.geometry import cart2sphere

def test_peak_directions_nl():
    def discrete_eval(sphere):
        return abs(sphere.vertices).sum(-1)

    directions, values = peak_directions_nl(discrete_eval)
    assert_equal(directions.shape, (4, 3))
    assert_array_almost_equal(abs(directions), 1/np.sqrt(3))
    assert_array_equal(values, abs(directions).sum(-1))

    # Test using a different sphere
    sphere = unit_icosahedron.subdivide(4)
    directions, values = peak_directions_nl(discrete_eval, sphere=sphere)
    assert_equal(directions.shape, (4, 3))
    assert_array_almost_equal(abs(directions), 1/np.sqrt(3))
    assert_array_equal(values, abs(directions).sum(-1))

    # Test the relative_peak_threshold
    def discrete_eval(sphere):
        A = abs(sphere.vertices).sum(-1)
        x, y, z = sphere.vertices.T
        B = 1 + (x*z > 0) + 2*(y*z > 0)
        return A * B

    directions, values = peak_directions_nl(discrete_eval, .01)
    assert_equal(directions.shape, (4, 3))

    directions, values = peak_directions_nl(discrete_eval, .3)
    assert_equal(directions.shape, (3, 3))

    directions, values = peak_directions_nl(discrete_eval, .6)
    assert_equal(directions.shape, (2, 3))

    directions, values = peak_directions_nl(discrete_eval, .8)
    assert_equal(directions.shape, (1, 3))
    assert_almost_equal(values, 4*3/np.sqrt(3))

    # Test odfs with large areas of zero
    def discrete_eval(sphere):
        A = abs(sphere.vertices).sum(-1)
        x, y, z = sphere.vertices.T
        B = (x*z > 0) + 2*(y*z > 0)
        return A * B

    directions, values = peak_directions_nl(discrete_eval, 0.)
    assert_equal(directions.shape, (3, 3))

    directions, values = peak_directions_nl(discrete_eval, .6)
    assert_equal(directions.shape, (2, 3))

    directions, values = peak_directions_nl(discrete_eval, .8)
    assert_equal(directions.shape, (1, 3))
    assert_almost_equal(values, 3*3/np.sqrt(3))

_sphere = create_unit_hemisphere(4)
_odf = (_sphere.vertices * [1, 2, 3]).sum(-1)
class SimpleOdfModel(OdfModel):
    sphere = _sphere
    def fit(self, data):
        fit = SimpleOdfFit()
        fit.model = self
        return fit

class SimpleOdfFit(OdfFit):
    def odf(self, sphere=None):
        if sphere is None:
            sphere = self.model.sphere

        # Use ascontiguousarray to work around a bug in NumPy
        return np.ascontiguousarray((sphere.vertices * [1, 2, 3]).sum(-1))

def test_OdfFit():
    m = SimpleOdfModel()
    f = m.fit(None)
    odf = f.odf(_sphere)
    assert_equal(len(odf), len(_sphere.theta))

def test_peak_directions():
    model = SimpleOdfModel()
    fit = model.fit(None)
    odf = fit.odf()

    argmax = odf.argmax()
    mx = odf.max()
    sphere = fit.model.sphere

    # Only one peak
    dir, val, ind = peak_directions(odf, sphere, .5, 45)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    assert_array_equal(dir, dir_e)

    odf[0] = mx * .9
    # Two peaks, relative_threshold
    dir, val, ind = peak_directions(odf, sphere, 1., 0)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    dir, val, ind = peak_directions(odf, sphere, .8, 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax, 0])
    assert_array_equal(val, odf[ind])

    # Two peaks, angle_sep
    dir, val, ind = peak_directions(odf, sphere, 0., 90)
    dir_e = sphere.vertices[[argmax]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax])
    assert_array_equal(val, odf[ind])
    dir, val, ind = peak_directions(odf, sphere, 0., 0)
    dir_e = sphere.vertices[[argmax, 0]]
    assert_array_equal(dir, dir_e)
    assert_array_equal(ind, [argmax, 0])
    assert_array_equal(val, odf[ind])

def test_peaksFromModel():
    data = np.zeros((10,2))

    # Test basic case
    model = SimpleOdfModel()
    odf_argmax = _odf.argmax()
    pam = peaks_from_model(model, data, _sphere, .5, 45, normalize_peaks=True)

    assert_array_equal(pam.gfa, gfa(_odf))
    assert_array_equal(pam.peak_values[:, 0], 1.)
    assert_array_equal(pam.peak_values[:, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[:, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[:, 1:], 0.)
    assert_array_equal(pam.peak_indices[:, 0], odf_argmax)
    assert_array_equal(pam.peak_indices[:, 1:], -1)

    # Test that odf array matches and is right shape
    pam = peaks_from_model(model, data, _sphere, .5, 45, return_odf=True)
    expected_shape = (len(data), len(_odf))
    assert_equal(pam.odf.shape, expected_shape)
    assert_true((_odf == pam.odf).all())
    assert_array_equal(pam.peak_values[:, 0], _odf.max())

    # Test mask
    mask = (np.arange(10) % 2) == 1

    pam = peaks_from_model(model, data, _sphere, .5, 45, mask=mask,
                           normalize_peaks=True)
    assert_array_equal(pam.gfa[~mask], 0)
    assert_array_equal(pam.qa[~mask], 0)
    assert_array_equal(pam.peak_values[~mask], 0)
    assert_array_equal(pam.peak_indices[~mask], -1)

    assert_array_equal(pam.gfa[mask], gfa(_odf))
    assert_array_equal(pam.peak_values[mask, 0], 1.)
    assert_array_equal(pam.peak_values[mask, 1:], 0.)
    mn, mx = _odf.min(), _odf.max()
    assert_array_equal(pam.qa[mask, 0], (mx - mn) / mx)
    assert_array_equal(pam.qa[mask, 1:], 0.)
    assert_array_equal(pam.peak_indices[mask, 0], odf_argmax)
    assert_array_equal(pam.peak_indices[mask, 1:], -1)

def test_sf_to_sh():
    #sphere = get_sphere('symmetric362')
    sphere = unit_octahedron
    sphere = sphere.subdivide(2)
    
    mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003] ))
    mevecs = [ np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ),
               np.array( [ [0,1,0], [1,0,0], [0,0,1] ] ) ]
    
    odf = multi_tensor_odf( sphere.vertices, [0.5, 0.5], mevals, mevecs )

    # 1D case with the 3 bases functions
    odf_sh = sf_to_sh(odf, sphere, 8)
    odf2 = sh_to_sf(odf_sh, sphere, 8)
    assert_array_almost_equal(odf, odf2, 2)
    
    odf_sh = sf_to_sh(odf, sphere, 8, "mrtrix")
    odf2 = sh_to_sf(odf_sh, sphere, 8, "mrtrix")
    assert_array_almost_equal(odf, odf2, 2)
        
    odf_sh = sf_to_sh(odf, sphere, 8, "fibernav")
    odf2 = sh_to_sf(odf_sh, sphere, 8, "fibernav")
    assert_array_almost_equal(odf, odf2, 2)

    # 2D case
    odf2d = np.vstack((odf2, odf))
    odf2d_sh = sf_to_sh(odf2d, sphere, 8)
    odf2d_sf = sh_to_sf(odf2d_sh, sphere, 8)
    assert_array_almost_equal(odf2d, odf2d_sf, 2)

def test_deconv():
    SNR = 20 #None #10, 20, 30
    bvalue = 1000
    S0 = 1
    sh_order = 8
    visu = False
    generate = True


    from dipy.data import get_data
    _, fbvals, fbvecs = get_data('small_64D')


    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]

    m, n = sph_harm_ind_list(sh_order)
    where_b0s = lazy_index(gtab.b0s_mask)
    where_dwi = lazy_index(~gtab.b0s_mask)
    x, y, z = gtab.gradients[where_dwi].T
    r, pol, azi = cart2sphere(x, y, z)
    B_dwi = real_sph_harm(m, n, azi[:, None], pol[:, None])

    
    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)

    psphere = get_sphere('symmetric362')    
    odf_gt = multi_tensor_odf(psphere.vertices, [0.5, 0.5], mevals, mevecs)

    r_bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    r_bvals = np.zeros(len(r_bvecs)) + bvalue
    r_bvals[0] = 0
    r_gtab = gradient_table(r_bvals, r_bvecs)
    R = single_tensor( r_gtab, S0, mevals[1], mevecs[1], snr=None )

    s_sh   = np.linalg.lstsq(B_dwi, S[1:])[0]
    r_sh,B_regul = sf_to_sh2( R[1:], psphere, sh_order )
    r_rh         = sh_to_rh( r_sh, sh_order )

    
#     if generate : 
#         print 'Done simulating signal...'    
#         # single fiber response function gtab on a sphere of 362 points
        

#         R = single_tensor( r_gtab, S0, s_mevals[1], s_mevecs[1], snr=None )
#         r_odf = single_tensor_odf( psphere.vertices, s_mevals[1], s_mevecs[1] )
#         print 'Done simulating response function...'
            
#         if visu :
#             from dipy.viz import fvtk
#             r = fvtk.ren()
#             fvtk.add( r, fvtk.sphere_funcs( np.vstack((R[1:], r_odf)), psphere ) )
#             fvtk.show( r )
                
#        np.savez('stuff_snr05.npz', B_dwi=B_dwi, B_regul=B_regul, s_sh=s_sh, r_sh=r_sh, r_rh=r_rh)
#    else :
#        npz = np.load('stuff_snr10.npz')
#        print npz.files
#        B_dwi = npz['B_dwi']
#        B_regul = npz['B_regul']
#        s_sh = npz['s_sh']
#        r_sh = npz['r_sh']
#        r_rh = npz['r_rh']

                
#     u_fodf_sh,b = sdeconv( r_rh, s_sh, sh_order, False )
#     u_fodf = sh_to_sf(u_fodf_sh, psphere, sh_order)

#     if visu :
#         odf_gt = multi_tensor_odf( psphere.vertices, [0.5, 0.5], s_mevals, s_mevecs )
#         from dipy.viz import fvtk
#         r = fvtk.ren()
#         fvtk.add( r, fvtk.sphere_funcs( np.vstack((odf_gt, u_fodf)), psphere ) )
#         fvtk.show( r )
        
#     f_fodf_sh,b = sdeconv( r_rh, s_sh, sh_order, True )
#     f_fodf = sh_to_sf(f_fodf_sh, psphere, sh_order)
    
#     if visu :
#         odf_gt = multi_tensor_odf( psphere.vertices, [0.5, 0.5], s_mevals, s_mevecs )
#         from dipy.viz import fvtk
#         r = fvtk.ren()
#         fvtk.add( r, fvtk.sphere_funcs( np.vstack((odf_gt, f_fodf)), psphere ) )
#         fvtk.show( r )



#     if visu :
#         odf_gt = multi_tensor_odf( psphere.vertices, [0.5, 0.5], s_mevals, s_mevecs )
#         from dipy.viz import fvtk
#         r = fvtk.ren()
#         fvtk.add( r, fvtk.sphere_funcs( np.vstack((odf_gt, fodf)), psphere ) )
#         fvtk.show( r )

    # building forward spherical signal convolution matrix
    b = np.zeros( (r_sh.shape) )
    bb = np.zeros( (r_sh.shape) )
    bbb = np.zeros( (r_sh.shape) )
    pp = np.zeros( (r_sh.shape) )
    i = 0
    for l in np.arange(0,sh_order+1,2) :
        for m in np.arange(-l,l+1) :
            b[i] = r_rh[l/2]
            i = i + 1
    R = np.diag( b )

    # building the forward and SDT convolution matrices 
    i = 0
    num = 1000
    delta = 1.0/num
    e1 = 15.0 # 13.9
    e2 = 3.0 #3.55
    ratio = e2/e1
    r = np.zeros( (r_rh.shape) )
    sdt = np.zeros( (r_rh.shape) )
    frt = np.zeros( (r_rh.shape) )

    for l in np.arange(0,sh_order+1,2) :
        sharp = 0.0
        integral = 0.0
        
        # Trapezoidal integration
        # 1/2 [ f(x0) + 2f(x1) + ... + 2f(x{n-1}) + f(xn) ] delta
        for z in np.linspace(-1, 1, num) :
            if z == -1 or z == 1  :
                sharp += lpn(l, z)[0][-1] * np.sqrt(1 / (1 - (1 - ratio) * z * z))
                integral += np.sqrt(1/(1 - (1 - ratio) * z * z))                    
            else  :
                sharp += 2 * lpn(l, z)[0][-1] * np.sqrt( 1 / (1 - (1 - ratio) * z * z))
                integral += 2 * np.sqrt(1/(1 - (1 - ratio) * z * z))
                
        integral /= 2
        integral *= delta
        sharp /= 2
        sharp *= delta
        sharp /= integral

        r[l/2] = 2 * np.pi * lpn(l, 0)[0][-1] / sharp
        sdt[l/2] = 1 / sharp
        frt[l/2] = 2 * np.pi * lpn(l, 0)[0][-1] 

    print 1/sdt    
    # This test is for e1 = 13.9 and e2 = 3.55 case    
    #r_bis = np.array([6.283, -36.829, 148.088, -537.100, 1828.486])
    #assert_array_almost_equal(r, r_bis, 0)
    # [ 1.          0.0987961   0.0214013   0.00570876  0.00169231]
    
    i = 0
    for l in np.arange(0,sh_order+1,2) :
        for m in np.arange(-l,l+1) :
            b[i] = r[l/2]
            bb[i] = frt[l/2]
            bbb[i] = sdt[l/2]
            pp[i] = 1/sdt[l/2]
            i = i + 1
    # P_frt is the Funk-Radon transform matrix
    P_frt = np.diag( bb )
    # P is the SDT matrix to go from S to fODF
    P = np.diag( b )
    # P_sdt is the SDT matrix to go from ODF to fODF
    P_sdt = np.diag( bbb )
    P_sdt_inv = np.diag( pp )



#    print np.dot(P_sdt, P_sdt_inv)

    
    odf_sh = np.dot( P_frt, s_sh )
    qball_odf = sh_to_sf(odf_sh, psphere, sh_order)
    Z = np.linalg.norm(qball_odf)
    odf_sh /= Z

    signal = sh_to_sf(s_sh, psphere, sh_order)
        
    #     fodf_sh = np.dot( P, s_sh )
    #     fodf = sh_to_sf(fodf_sh, psphere, sh_order)
    
    #     fodf_sh2 = np.dot( P_sdt, odf_sh )
    #     fodf2 = sh_to_sf(fodf_sh, psphere, sh_order)
    
    #     fodf_sh3 = np.linalg.lstsq(P_sdt_inv, odf_sh)[0]  
    #     fodf3 = sh_to_sf(fodf_sh3, psphere, sh_order)
    
    #     odf_sh2 = np.dot( P_sdt_inv, fodf_sh2 )
    #     odf2 = sh_to_sf(odf_sh2, psphere, sh_order)
    
    #     odf_sh3 = np.dot( P_sdt_inv, fodf_sh )
    #     odf3 = sh_to_sf(odf_sh2, psphere, sh_order)
    
    if visu :
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add( r, fvtk.sphere_funcs( np.vstack((qball_odf, signal)), psphere ) )
        fvtk.show( r )

    fodf_sh,num_it = csdeconv( r_rh, s_sh, sh_order, B_regul, 1.0, 0.1 )
    fodf_csd = sh_to_sf(fodf_sh, psphere, sh_order)
    print 'CSD converged after %d iterations'%num_it

    fodf_sh,num_it = odf_csdeconv_usingS( s_sh, sh_order, R, P, B_regul, 1, 0.1 )
    fodf_csd_odf = sh_to_sf(fodf_sh, psphere, sh_order)
    print 'CSD-ODF converged after %d iterations'%num_it

    # odf_sh should be normalized
    fodf_csd_sh,num_it = odf_deconv( odf_sh, sh_order, P_sdt_inv, B_regul, 1, 0.025 )
    fodf_sdt = sh_to_sf(fodf_csd_sh, psphere, sh_order)
    print 'SDT CSD converged after %d iterations'%num_it
    
    visu = True
    if visu :
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add( r, fvtk.sphere_funcs( np.vstack((fodf_csd, fodf_csd_odf, fodf_sdt)), psphere ) )
        fvtk.show( r )


def odf_deconv( odf_sh, sh_order, P_sdt_inv, B_regul, Lambda, tau ) :
    """ ODF constrained-regularized sherical deconvolution

    Parameters
    ----------
    odf_sh : ndarray
         ndarray of SH coefficients for the ODF spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    P_sdt : ndarray
         SDT matrix in SH basis
    B_regul : ndarray
         SH basis matrix used for deconvolution
    Lambda : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         tau parameter in the L matrix construction (default 0.1)
         
    Returns
    _______
    fodf_sh : ndarray
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    Descoteaux, M, et. al. TMI 2009.
    """
    m, n = sph_harm_ind_list(sh_order)

    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.linalg.lstsq(P_sdt_inv, odf_sh)[0]  
    fodf_sh[15:] = 0

    fodf = np.dot(B_regul,fodf_sh)
    Z = np.linalg.norm(fodf)
    fodf_sh /= Z

    # tau should be more or less around 0.025 from my experience
    # a good heuristic choice is just the mean of the fodf on the sphere. 
    threshold=tau
    #print np.mean(np.dot(B_regul, fodf_sh))
    threshold = np.mean(np.dot(B_regul, fodf_sh))
    
    Lambda = Lambda * P_sdt_inv.shape[0] * P_sdt_inv[0,0] / B_regul.shape[0]
    
    #print Lambda,threshold
    
    k = []
    convergence = 50
    for num_it in np.arange(1,convergence+1) :
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero( A < threshold )[0]
        
        if (k2.shape[0] + P_sdt_inv.shape[0])  < B_regul.shape[1] :
            print 'too few negative directions identified - failed to converge'
            return fodf_sh,num_it
    
        if num_it > 1 and k.shape[0] == k2.shape[0] :
            if (k == k2).all() :
                return fodf_sh,num_it
           
        k = k2
        M = np.concatenate( (P_sdt_inv, Lambda*B_regul[k, :] ) )
        ODF = np.concatenate( (odf_sh, np.zeros( k.shape ) ) )
        fodf_sh = np.linalg.lstsq(M, ODF)[0]  # M\ODF        
        
    print 'maximum number of iterations exceeded - failed to converge';
    return fodf_sh,num_it


def odf_csdeconv_usingS( s_sh, sh_order, R, P, B_regul, Lambda, tau ) :
    """ ODF constrained-regularized sherical deconvolution       

    Parameters
    ----------
    s_sh : ndarray
         ndarray of SH coefficients for the spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    R : ndarray
         Single fiber response function SH matrix
    P : ndarray
         SDT matrix in SH basis
    B_regul : ndarray
         SH basis matrix used for deconvolution
    Lambda : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         tau parameter in the L matrix construction (default 0.1)
         
    Returns
    _______
    fodf_sh : ndarray
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    Descoteaux, M, et. al. TMI 2009.
    """
    m, n = sph_harm_ind_list(sh_order)

    # Generate initial fODF estimate, which is the ODF truncated at SH order 4
    fodf_sh = np.dot( P, s_sh)
    fodf_sh[15:] = 0

    #set threshold on FOD amplitude used to identify 'negative' values
    #threshold = tau
    fodf = np.dot(B_regul, fodf_sh)
    #print np.mean(fodf),np.linalg.norm(fodf),np.min(fodf),np.max(fodf)
    Z = np.linalg.norm(fodf)
    fodf_sh /= Z
    
    threshold = tau*np.mean(np.dot(B_regul, fodf_sh));
    Lambda = Lambda * R.shape[0] * R[0,0] / B_regul.shape[0]
    #print Lambda,threshold
    
    k = []
    convergence = 50
    for num_it in np.arange(1,convergence+1) :
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero( A < threshold )[0]
        
        if (k2.shape[0] + R.shape[0])  < B_regul.shape[1] :
            print 'too few negative directions identified - failed to converge'
            return fodf_sh,num_it
    
        if num_it > 1 and k.shape[0] == k2.shape[0] :
            if (k == k2).all() :
                return fodf_sh,num_it
           
        k = k2
        M = np.concatenate( (R, Lambda*B_regul[k, :] ) )
        S = np.concatenate( (s_sh, np.zeros( k.shape ) ) )
        fodf_sh = np.linalg.lstsq(M, S)[0]  # M\S
        
    print 'maximum number of iterations exceeded - failed to converge';
    return fodf_sh,num_it


def csdeconv( r_rh, s_sh, sh_order, B_regul, Lambda, tau ) :
    """ Constrained-regularized spherical deconvolution (CSD)

    Deconvolves the axially symmetric single fiber response
    function `r_rh` in rotational harmonics coefficients from the spherical function
    `s_sh` in SH coefficients. 
    
    Parameters
    ----------
    r_rh : ndarray
         ndarray of rotational harmonics coefficients for the single fiber response function
    s_sh : ndarray
         ndarray of SH coefficients for the spherical function to be deconvolved
    sh_order : int
         maximal SH order of the SH representation
    B_regul : ndarray
         SH basis matrix used for deconvolution
    Lambda : float
         lambda parameter in minimization equation (default 1.0)
    tau : float
         tau parameter in the L matrix construction (default 0.1)
         
    Returns
    _______
    fodf_sh : ndarray
         Spherical harmonics coefficients of the constrained-regularized fiber ODF
    num_it : int
         Number of iterations in the constrained-regularization used for convergence

    References
    ----------
    Tournier, J.D., et. al. NeuroImage 2007.
    """
    m, n = sph_harm_ind_list(sh_order)

    # building forward spherical deconvolution matrix
    b = np.zeros( (m.shape) )
    i = 0
    for l in np.arange(0,sh_order+1,2) :
        for m in np.arange(-l,l+1) :
            b[i] = r_rh[l/2]                
            i = i + 1
    R = np.diag( b )

    # generate initial fODF estimate, truncated at SH order 4
    fodf_sh = np.linalg.lstsq(R, s_sh)[0]  # R\sh_sh
    fodf_sh[15:] = 0    
    psphere = get_sphere('symmetric362')
    fodf = sh_to_sf(fodf_sh, psphere, sh_order)
    
    #set threshold on fODF amplitude used to identify 'negative' values
    threshold = tau*np.mean(np.dot(B_regul, fodf_sh));

    # scale lambda to account for differences in the number of 
    # SH coefficients and number of mapped directions
    Lambda = Lambda * R.shape[0] * r_rh[0] / B_regul.shape[0]

    #print threshold,Lambda
    #print R.shape[0], r_rh[0], B_regul.shape[0]
    
    if sh_order == 8 :
        assert_almost_equal(Lambda, 0.822627111014, 4) # as in mrtrix

    k = []
    convergence = 50
    for num_it in np.arange(1,convergence+1) :
        A = np.dot(B_regul, fodf_sh)
        k2 = np.nonzero( A < threshold )[0]
        
        if (k2.shape[0] + R.shape[0])  < B_regul.shape[1] :
            print 'too few negative directions identified - failed to converge'
            return fodf_sh,num_it
    
        if num_it > 1 and k.shape[0] == k2.shape[0] :
            if (k == k2).all() :
                return fodf_sh,num_it
           
        k = k2
        M = np.concatenate( (R, Lambda*B_regul[k, :] ) )
        S = np.concatenate( (s_sh, np.zeros( k.shape ) ) )
        fodf_sh = np.linalg.lstsq(M, S)[0]  # M\S
        
    print 'maximum number of iterations exceeded - failed to converge';
    return fodf_sh,num_it


def sdeconv( r_rh, s_sh, sh_order, filter ) :
    """ Unfiltered and filtered spherical deconvolution 

    Deconvolves the axially symmetric single fiber response
    function `r_rh` in rotational harmonics coefficients from
    the spherical function `s_sh` in SH coefficients. 

    If filter is True, perform 'standard' filtered
    spherical deconvolution. Note that the
    filtering coefficients are supplied as their reciprocals - the 
    filter coefficients here are actually
    [ 1 1 1 0.1 0.01 inf inf inf ] 
    (the last four correspond to sh_order > 8 and are unused)

    Parameters
    ----------
    r_rh : ndarray
         ndarray of rotational harmonics coefficients for
         the single fiber response function
    s_sh : ndarray
         ndarray of spherical harmonics coefficients for
         the spherical function to deconvolve
    sh_order : int
         maximal SH order of the SH representation
    filted : bool
         Filter or not the fiber ODF coefficients output
         
    Returns
    _______
    fodf_sh : ndarray
         Spherical harmonics coefficients of the unfiltered fiber ODF

    References
    ----------
    Tournier, J.D., et. al. NeuroImage 2004.
    """
    m, n = sph_harm_ind_list(sh_order)
    # other bases support TO DO

    if filter :
        f = np.zeros( (r_rh.shape) )
        f[:5] = np.array([1, 1, 1, 2, 10])
        r_rh = r_rh*f
    
    b = np.zeros( (m.shape) )
    i = 0
    for l in np.arange(0,sh_order+1,2) :
        for m in np.arange(-l,l+1) :
            b[i] = 1/r_rh[l/2]                
            i = i + 1

    fodf_sh = b * s_sh        
    return fodf_sh,b


def test_b() :
    # this values come from mrtrix
    r_rh = np.array( [1.7450, -0.6124, 0.2205, -0.0657, 0.01622] )
    b = np.array( [0.5731, -1.6329, 4.5345, -15.2235, 61.6530 ] )    
    s_sh = np.zeros( (45) )
    fodf_sh,b_bis = sdeconv(r_rh, s_sh, 8, False)
    k = np.array( [0, 3, 10, 21, 36] )
    assert_array_almost_equal(b, b_bis[k], 2 )
    
    
def sh_to_rh( r_sh, sh_order ) :
    """ Spherical harmonics (SH) to rotational harmonics (RH)

    Calculate the rotational harmonic decomposition up to 
    harmonic sh_order for an axially and antipodally 
    symmetric function. Note that
    all m != 0 coefficients will be ignored as axial symmetry
    is assumed.
    
    Parameters
    ----------
    r_sh : ndarray
         ndarray of SH coefficients for the single fiber response function 
    sh_order : int
         maximal SH order of the SH representation
         
    Returns
    _______
    r_rh : ndarray
         Rotational harmonics coefficients representing the input `r_sh`
    """

    dirac_sh = gen_dirac(0, 0, sh_order)
    k = np.nonzero( dirac_sh )[0]
    r_rh = r_sh[k] / dirac_sh[k]
    return r_rh
    
def gen_dirac( pol, azi, sh_order ) :
    m, n = sph_harm_ind_list(sh_order)
    rsh = real_sph_harm
    # other bases support TO DO
    
    dirac = np.zeros( (m.shape) )
    i = 0
    for l in np.arange(0,sh_order+1,2) :
        for m in np.arange(-l,l+1) :
            if m == 0 :
                dirac[i] = rsh(0, l, azi, pol)

            i = i + 1

    return dirac

def test_gen_dirac() :
    dirac = np.array([0.2821, 0, 0, 0.6308, 0, 0, 0, 0, 0, 0, 0.8463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1631, 0, 0, 0, 0, 0, 0, 0, 0])
    # as from mrtrix 
    dirac2 = gen_dirac(0, 0, 8)
    assert_array_almost_equal(dirac, dirac2, 2 )

        
def sf_to_sh2(sf, sphere, sh_order=4, basis_type=None, smooth=0.0):
    """ Spherical function to spherical harmonics (SH)
    
    Parameters
    ----------
    sf : ndarray
         ndarray of values representing spherical functions on the 'sphere'
    sphere : Sphere
          The points on which the sf is defined.
    sh_order : int, optional
               Maximum SH order in the SH fit,
               For `sh_order`, there will be
               (`sh_order`+1)(`sh_order`_2)/2 SH coefficients
               (default 4)
    basis_type : {None, 'mrtrix', 'fibernav'}
                 None for the default dipy basis,
                 'mrtrix' for the MRtrix basis, and
                 'fibernav' for the FiberNavigator basis
                 (default None)
   smooth : float, optional
            Lambda-regularization in the SH fit
            (default 0.0)
    
    Returns
    _______
    sh : ndarray
         SH coefficients representing the input `odf`
    B :  ndarray
         SH basis matrix, N x (`sh_order`+1)(`sh_order`_2)/2
         N is the number of points on the `sphere`
             
    """
    m, n = sph_harm_ind_list(sh_order)

    pol = sphere.theta
    azi = sphere.phi

    sph_harm_basis = sph_harm_lookup.get(basis_type)
    if not sph_harm_basis:
        raise ValueError(' Wrong basis type name ')
    B = sph_harm_basis(m, n, azi[:, None], pol[:, None])
    
    L = -n * (n + 1)
    invB = smooth_pinv(B, np.sqrt(smooth)*L)
    R = (sh_order + 1) * (sh_order + 2) / 2
    sh = np.zeros(sf.shape[:-1] + (R,))

    sh = np.dot(sf, invB.T)        

    return sh,B



#test_gen_dirac()
#test_b()
test_deconv()


    

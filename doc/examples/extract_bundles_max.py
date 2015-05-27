"""
===========================
Automatic bundle extraction
===========================
"""
import numpy as np
from os.path import basename
import nibabel as nib
import nibabel.trackvis as tv
from glob import glob
from dipy.viz import fvtk
from time import time, sleep

from dipy.tracking.streamline import (length,
                                      transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from itertools import chain, izip
from dipy.align.streamlinear import StreamlineLinearRegistration
from axycolor import distinguishable_colormap
from os import mkdir
from os.path import isdir
import os
from dipy.io.pickles import load_pickle, save_pickle
import tractconverter as tc
from dipy.tracking.streamline import select_random_set_of_streamlines


def read_fib(fname, random_N=100000):    
    tracts_format = tc.detect_format(fname)
    tracts_file = tracts_format(fname)

    strls = [s for s in tracts_file]

    if random_N is not None :
        streams = select_random_set_of_streamlines(strls, random_N)
    else :
        streams = strls
    
    return streams


def show_bundles(static, moving, linewidth=1., tubes=False,
                 opacity=1., fname=None):
    ren = fvtk.ren()
    ren.SetBackground(1, 1, 1.)

    if tubes:
        static_actor = fvtk.streamtube(static, fvtk.colors.red,
                                       linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.streamtube(moving, fvtk.colors.green,
                                       linewidth=linewidth, opacity=opacity)

    else:
        static_actor = fvtk.line(static, fvtk.colors.red,
                                 linewidth=linewidth, opacity=opacity)
        moving_actor = fvtk.line(moving, fvtk.colors.green,
                                 linewidth=linewidth, opacity=opacity)

    fvtk.add(ren, static_actor)
    fvtk.add(ren, moving_actor)

    fvtk.add(ren, fvtk.axes(scale=(2, 2, 2)))

    fvtk.show(ren, size=(900, 900))
    if fname is not None:
        fvtk.record(ren, size=(900, 900), out_path=fname)


def ismrm_next_bundle(model_bundles_dir, verbose=False):

    if verbose :
        print 'Model dir:', model_bundles_dir

    if False :
        wb_trk2 = model_bundles_dir + 'IOFF_left.fib'
        wb2 = read_fib(wb_trk2, None)
        tag = basename(wb_trk2).split('.fib')[0]
        yield (wb2, tag)

    else :
        for wb_trk2 in glob(model_bundles_dir + '*.fib'):
            if verbose:
                print(wb_trk2)
                
            wb2 = read_fib(wb_trk2, None)
        
            tag = basename(wb_trk2).split('.fib')[0]
        
            if verbose:
                print(tag)

            yield (wb2, tag)


def ismrm_initial(full_tracts_dir, fulldataset_name, fiber_extension, 
                  random_N, verbose=False):
    
    full_brain_tracts = full_tracts_dir + fulldataset_name + fiber_extension

    if verbose :
        print(full_brain_tracts)


    streamlines = read_fib(full_brain_tracts, random_N)

    return streamlines



def auto_extract(model_bundle, moved_streamlines,
                 number_pts_per_str=12, 
                 close_centroids_thr=20,
                 clean_thr=7.,
                 disp=False, verbose=True, expand_thr=None):

    if verbose:
        print('# Centroids of model bundle')

    t0 = time()

    rmodel_bundle = set_number_of_points(model_bundle, number_pts_per_str)
    rmodel_bundle = [s.astype('f4') for s in rmodel_bundle]

    qb = QuickBundles(threshold=20)
    model_cluster_map = qb.cluster(rmodel_bundle)
    model_centroids = model_cluster_map.centroids

    if verbose:
        print('Duration %f ' % (time() - t0, ))

    if verbose:
        print('# Calculate centroids of moved_streamlines')

    t = time()

    rstreamlines = set_number_of_points(moved_streamlines, number_pts_per_str)
    # qb.cluster had problem with f8
    rstreamlines = [s.astype('f4') for s in rstreamlines]

    cluster_map = qb.cluster(rstreamlines)
    cluster_map.refdata = moved_streamlines

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if verbose:
        print('# Find centroids which are close to the model_centroids')

    t = time()

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf

    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]

    #close_centroids = [cluster.centroid for cluster in close_clusters]

    close_streamlines = list(chain(*close_clusters))

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if disp and False:
        show_bundles(model_bundle, close_streamlines)

    closer_streamlines = close_streamlines
    matrix = np.eye(4)

    if verbose:
        print('# Remove streamlines which are a bit far')

    t = time()

    rcloser_streamlines = set_number_of_points(closer_streamlines, number_pts_per_str)

    clean_matrix = bundles_distances_mdf(rmodel_bundle, rcloser_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)
    close_clusters_clean = [closer_streamlines[i]
                            for i in np.where(mins != np.inf)[0]]

    if verbose:
        print('Duration %f ' % (time() - t, ))

    msg = 'Total duration of automatic extraction %0.4f seconds.'
    print(msg % (time() - t0, ))
    if disp:
        show_bundles(model_bundle, close_clusters_clean)

    if expand_thr is not None:
        rclose_clusters_clean = set_number_of_points(close_clusters_clean, number_pts_per_str)
        expand_matrix = bundles_distances_mam(rclose_clusters_clean,
                                              rcloser_streamlines)

        expand_matrix[expand_matrix > expand_thr] = np.inf
        mins = np.min(expand_matrix, axis=0)
        expanded = [closer_streamlines[i]
                    for i in np.where(mins != np.inf)[0]]

        return expanded, matrix

    return close_clusters_clean, matrix



def exp_validation_with_ismrm(model_tracts_dir, 
                              full_tracts_dir, 
                              fiber_extension, 
                              full_brain_streamlines_tag,
                              random_N, 
                              number_pts_per_str=12, 
                              close_centroids_thr=20,
                              clean_thr=5.,
                              verbose=False,
                              disp=False,
                              expand_thr=None):

    print(full_brain_streamlines_tag)    
    full_streamlines = ismrm_initial(full_tracts_dir, full_brain_streamlines_tag, 
                                     fiber_extension, random_N=random_N, verbose=verbose)

    for (streamlines, tag) in ismrm_next_bundle(model_tracts_dir):
        print(tag)
        t = time()

        print 'Full streamlines number:', len(full_streamlines)
        print 'model streamlines number:', len(streamlines)

        if disp :
            show_bundles(streamlines, full_streamlines)
        
        # Big bundles with many branches can handle larger clean_thr
        # CA and CP go through a 1-2 voxels funnel, they need a small clean_thr
        # Others are stable at 5-7 
        if tag == 'CP' or tag == 'CA':
            new_clean_thr = 2.
            print 'For bundle', tag, 'clean threshold is:', new_clean_thr
        if tag == 'Fornix' or tag=='MCP' or tag=='SLF_left' or tag=='SLF_right' or tag=='CC' or tag=='Cingulum_left' or tag=='Cingulum_right':
            new_clean_thr = 10.
            print 'For bundle', tag, 'clean threshold is:', new_clean_thr
        if tag == 'ICP_left' or tag == 'ICP_right' :
            new_clean_thr = 6.
            print 'For bundle', tag, 'clean threshold is:', new_clean_thr
        else :
            new_clean_thr = clean_thr
            print 'For bundle', tag, 'clean threshold is:', new_clean_thr
        
        extracted, mat2 = auto_extract(streamlines, full_streamlines,
                                       number_pts_per_str=number_pts_per_str,
                                       close_centroids_thr=close_centroids_thr,
                                       clean_thr=new_clean_thr,
                                       disp=disp, verbose=verbose,
                                       expand_thr=expand_thr)

        if len(extracted) > 0 :
            print len(extracted), ' streamlines found...'
            show_bundles(streamlines, extracted)
        else :
            print 'The dataset does not contain this bundle...'

        # here we need to write as we want
        


if __name__ == '__main__':

    model_tracts_dir = '/home/local/USHERBROOKE/desm2239/Research/Data/Challenge/Final/tracts/'
    full_tracts_dir = '/home/local/USHERBROOKE/desm2239/Research/Data/Challenge/full_datasets/'
    fiber_extension = '.fib'
    
    files = ['3_0', '4_0', '10_10', '11_0', '14_0', '20_0'];

    exp_validation_with_ismrm(model_tracts_dir,
                              full_tracts_dir,
                              fiber_extension, 
                              full_brain_streamlines_tag=files[0], 
                              random_N=100000, # randomly pick N fibers from the full set of streamline. 
                              number_pts_per_str=12, # this is low but works fine
                              close_centroids_thr=20,
                              clean_thr=7., # 7 seems like a good compromise. Larger than that we add spurious tracts
                              verbose=False,
                              disp=False,
                              expand_thr=None) 



# perfect overlap for 'merged_final_bundles'




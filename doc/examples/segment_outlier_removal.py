import sys
import os
import readline
import numpy as np
import pylab as plt
import nibabel as nib

from time import time

from scipy.special import ndtri

from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles, Cluster
from itertools import takewhile, count, izip, chain


def get_streamlines_bounding_box(streamlines):
    box_min = np.array([np.inf, np.inf, np.inf])
    box_max = -np.array([np.inf, np.inf, np.inf])

    for s in streamlines:
        box_min = np.minimum(box_min, np.min(s, axis=0))
        box_max = np.maximum(box_max, np.max(s, axis=0))

    return box_min, box_max


def show_clusters(clusters, colormap=None, cam_pos=None,
                  cam_focal=None, cam_view=None,
                  magnification=1, fname=None, size=(900, 900)):

    from dipy.viz import fvtk
    from dipy.viz.axycolor import distinguishable_colormap
    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)
    for cluster, color in izip(clusters, colormap):
        fvtk.add(ren, fvtk.line(list(cluster), [color]*len(cluster), linewidth=3))

    fvtk.show(ren, size=size)
    cam = ren.GetActiveCamera()
    return cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()


def show_clusters_grid_view(clusters, colormap=None, makelabel=None, grid_of_clusters=False,
                            cam_pos=None, cam_focal=None, cam_view=None,
                            magnification=1, fname=None, size=(900, 900), oname=''):

    from dipy.viz import fvtk
    from dipy.viz.axycolor import distinguishable_colormap

    def grid_distribution(N):
        def middle_divisors(n):
            for i in range(int(n ** (0.5)), 2, -1):
                if n % i == 0:
                    return i, n // i

            return middle_divisors(n+3)  # If prime number take next one

        height, width = middle_divisors(N)
        X, Y, Z = np.meshgrid(np.arange(width), np.arange(height), [0])
        return np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    bg = (1, 1, 1)
    if colormap is None:
        colormap = distinguishable_colormap(bg=bg)

    positions = grid_distribution(len(clusters))

    if grid_of_clusters:
        box_min, box_max = get_streamlines_bounding_box(chain(*chain(*clusters)))
    else:
        box_min, box_max = get_streamlines_bounding_box(chain(*clusters))

    ren = fvtk.ren()
    fvtk.clear(ren)
    ren.SetBackground(*bg)

    width, height, depth = box_max - box_min
    text_scale = [height*0.2]*3
    for cluster, color, pos in izip(clusters, colormap, positions):
        offset = pos * (box_max - box_min)
        offset[0] += pos[0] * 4*text_scale[0]
        offset[1] += pos[1] * 4*text_scale[1]

        if grid_of_clusters:
            for cc, color in izip(cluster, distinguishable_colormap(bg=bg)):
                fvtk.add(ren, fvtk.line([s + offset for s in cc], [color]*len(cc), linewidth=3))
        else:
            fvtk.add(ren, fvtk.line([s + offset for s in cluster], [color]*len(cluster), linewidth=3))

        if makelabel is not None:
            label = makelabel(cluster)
            text_pos = offset + np.array([-len(label)*text_scale[0]/2., height/2.+text_scale[1], 0])

            fvtk.label(ren, text=label, pos=text_pos, scale=text_scale, color=(0, 0, 0))

    if oname == '':
        fvtk.show(ren, size=size)
    else :
        fvtk.record(ren, out_path=oname, size=size)


def prune(streamlines, threshold, features, refdata=None):
    indices = np.arange(len(streamlines))

    outlier_indices = indices[features < threshold]
    rest_indices = indices[features >= threshold]

    if refdata is not None:
        # Redo QB for vizu on outliers and streamlines we keep
        qb_vizu = QuickBundles(threshold=5)
        clusters_outliers = qb_vizu.cluster(streamlines, ordering=outlier_indices, refdata=refdata)
        clusters_rest = qb_vizu.cluster(streamlines, ordering=rest_indices, refdata=refdata)

        outliers_cluster = Cluster(indices=outlier_indices, refdata=refdata)
        rest_cluster = Cluster(indices=rest_indices, refdata=refdata)

        # Show outliers streamlines vs. the ones we kept
        if len(outlier_indices) > 0 and len(rest_indices) > 0:
            show_clusters_grid_view([clusters_outliers, [outliers_cluster, rest_cluster], clusters_rest], 
                                    grid_of_clusters=True)
        else:
            show_clusters_grid_view([clusters_outliers, clusters_rest], grid_of_clusters=True)

    return outlier_indices, rest_indices


def outliers_removal_using_hierarchical_quickbundles(streamlines, confidence=0.95, 
                                                     min_threshold=0.5, nb_samplings_max=30, debug=False):
    sterror_factor = ndtri(confidence)
    metric = "mdf"

    box_min, box_max = get_streamlines_bounding_box(streamlines)
    #initial_threshold = np.sqrt(np.sum((box_max - box_min)**2)) / 4.  
    # Half of the bounding box's halved diagonal length.
    initial_threshold = np.min(np.abs(box_max - box_min)) / 2.

    # Quickbundle's threshold is halved between hierarchical level.
    thresholds = list(takewhile(lambda t: t >= min_threshold, (initial_threshold / 1.2**i for i in count())))

    start_time = time()
    ordering = np.arange(len(streamlines))
    nb_clusterings = 0

    streamlines_path = np.ones((len(streamlines), len(thresholds), nb_samplings_max), dtype=int) * -1
    for i in range(nb_samplings_max):
        np.random.shuffle(ordering)

        cluster_orderings = [ordering]
        for j, threshold in enumerate(thresholds):
            id_cluster = 0
            if debug :
                print "Ordering #{0}, QB/{2}mm, {1} clusters to process".format(i+1, len(cluster_orderings), 
                                                                                threshold)

            next_cluster_orderings = []
            qb = QuickBundles(metric=metric, threshold=threshold)
            for cluster_ordering in cluster_orderings:
                clusters = qb.cluster(streamlines, ordering=cluster_ordering)
                nb_clusterings += 1

                for k, cluster in enumerate(clusters):
                    streamlines_path[cluster.indices, j, i] = id_cluster
                    id_cluster += 1
                    if len(cluster) > 10:
                        next_cluster_orderings.append(cluster.indices)

            cluster_orderings = next_cluster_orderings

        if debug :
            print "{} qb done in {:.2f} sec on {} streamlines".format(nb_clusterings, time()-start_time, 
                                                                      len(streamlines))

        #path_lengths_per_streamline = np.sum(T[:, None]*(streamlines_path == -1), axis=1)
        path_lengths_per_streamline = np.sum((streamlines_path != -1), axis=1)[:, :i]

        # Compute confidence interval on mean cluster's size for each streamlines
        sterror_path_length_per_streamline = np.std(path_lengths_per_streamline, axis=1, ddof=1) / np.sqrt(i+1)
        if debug :
            print "Avg. sterror:", sterror_factor*sterror_path_length_per_streamline.mean()
            print "Max. sterror:", sterror_factor*sterror_path_length_per_streamline.max()

        if sterror_factor*sterror_path_length_per_streamline.mean() < 0.5:
            break

    summary = np.mean(path_lengths_per_streamline, axis=1) / np.max(path_lengths_per_streamline)
    return summary


def apply_on_specific_bundle(streamlines, confidence, alpha, oname1, oname2):
    from dipy.viz import fvtk
    rstreamlines = set_number_of_points(streamlines, 20)

    summary = outliers_removal_using_hierarchical_quickbundles(rstreamlines, confidence=confidence, debug=False)
    #summary = automatic_outliers_removal_inception_proba(Rstreamlines, vizu_qb.metric, agressivity=alpha, confidence=0.95, nb_thresholds_max=20, nb_samplings_max=nb_samplings_max)

    # something between 0.15 and 0.3 seems reasonable
    outliers, rest = prune(rstreamlines, alpha, summary)

    outliers_cluster = Cluster(indices=outliers, refdata=streamlines)
    rest_cluster = Cluster(indices=rest, refdata=streamlines)

    show_clusters_grid_view([rest_cluster, outliers_cluster], 
                            colormap=[fvtk.colors.green, fvtk.colors.orange_red],
                            oname=oname1)

    plt.hist(summary, bins=100)
    plt.savefig(oname2)
    plt.close()

    # this part is for the interactive mode
    interactive = True
    if interactive :
        plt.hist(summary, bins=100)
        plt.show(False)

        while True:
            print "---\nNew prunning threshold:",
            alpha = float(raw_input())

            outliers, rest = prune(rstreamlines, alpha, summary)
            print "Pruned {0} out of {1} streamlines at {2:.2f}%".format(len(outliers), 
                                                                         len(streamlines), alpha*100)

            outliers_cluster = Cluster(indices=outliers, refdata=streamlines)
            rest_cluster = Cluster(indices=rest, refdata=streamlines)
            show_clusters_grid_view([rest_cluster, outliers_cluster], 
                                    colormap=[fvtk.colors.green, fvtk.colors.orange_red])
            show_clusters([rest_cluster, outliers_cluster], colormap=[fvtk.colors.green, fvtk.colors.orange_red])
            show_clusters([rest_cluster], colormap=[fvtk.colors.orange_red])

    return rest_cluster, outliers_cluster


def load_specific_bundle(bundlename):
    #dname = '/Users/desm2239/Research/Data/Cunnane/Az_prob/09-043-0002-RR/work/results/'
    #fname = dname + 'bundles_{}.trk'.format(bundlename)
    fname = bundlename

    streams, hdr = nib.trackvis.read(fname)
    streamlines = [i[0] for i in streams]
    colors = [i[1] for i in streams]
    properties = [i[2] for i in streams]
    return streamlines, colors, properties, fname, hdr


def _hsv_to_rgb(h, s, v):
    h_i = int(h*6)
    f = h*6 - h_i
    p = v * (1-s)
    q = v * (1-f*s)
    t = v * (1 - (1-f) * s)
    if h_i == 0:
        r, g, b = v, t, p
    if h_i == 1:
        r, g, b = q, v, p
    if h_i == 2:
        r, g, b = p, v, t
    if h_i == 3:
        r, g, b = p, q, v
    if h_i == 4:
        r, g, b = t, p, v
    if h_i == 5:
        r, g, b = v, p, q
    return [int(r*256), int(g*256), int(b*256)]

def save_streamlines_with_color(streamlines, color, out_file, hdr):
    trk = []
    properties = None
    for i, streamline in enumerate(streamlines):
        scalars = np.array([color] * len(streamline))
        trk.append((streamline, scalars, properties))

    nib.trackvis.write(out_file, trk, hdr)

def save_specific_cluster(cluster, fname, hdr):
    streamlines = list(cluster)
    
    new_hdr = nib.trackvis.empty_header()
    new_hdr['voxel_size'] = hdr['voxel_size']
    new_hdr['voxel_order'] = hdr['voxel_order']
    new_hdr['dim'] = hdr['dim']
    new_hdr['n_count'] = len(streamlines)

    colormap = _hsv_to_rgb(0.5, 0.99, 0.99)
    
    # this is a hack because I don't know how to properly save streamlines...
    save_streamlines_with_color(streamlines, colormap, fname, new_hdr)        


if __name__ == '__main__':
    np.random.seed(43)

    confidence = 0.95
    if len(sys.argv) > 2:
        confidence = float(sys.argv[2])
    
    alpha = 0.2
    if len(sys.argv) > 3:
        alpha = float(sys.argv[3])


    streamlines, colors, properties, in_name, hdr = load_specific_bundle(sys.argv[1])
    
    fileName, fileExtension = os.path.splitext(in_name)
    #print(fileName, fileExtension)    
    dirname = os.path.split(os.path.abspath(in_name))[0]
    filename = os.path.split(os.path.abspath(in_name))[1]
    fileNameNoExt, fileExtension = os.path.splitext(filename)    
    oname2 = dirname + '/outliers/' + fileNameNoExt + '_outliers_hist.png'
    oname1 = dirname + '/outliers/' + fileNameNoExt + '_outliers.png'

    rest_cluster, outliers_cluster = apply_on_specific_bundle(streamlines, confidence, alpha, 
                                                              oname1, oname2)        
    

    from subprocess import call
    call(["mkdir", "outliers"])
    
    # how do we get the color of the actual TRK to save the same color?
    save_specific_cluster(rest_cluster, 
                          dirname + '/outliers/' + fileNameNoExt + '_clean' + fileExtension, hdr)
    save_specific_cluster(outliers_cluster, 
                          dirname + '/outliers/' + fileNameNoExt + '_outliers' + fileExtension, hdr)
    
    
    
    
    

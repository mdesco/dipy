from __future__ import division

from itertools import izip

from dipy.viz import window, actor, utils
from dipy.viz.colormap import distinguishable_colormap
from dipy.viz import interactor
from dipy.viz.utils import auto_orient

import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib
import sys
import glob

verbose=False
def main():
    print(glob.glob(sys.argv[1]))
    #print(glob.glob(sys.argv[2]))

    streamlines = []
    for tracts in glob.glob(sys.argv[1]) :        
        if verbose :
            print(tracts)
        streamlines += [nib.streamlines.load(tracts).streamlines]

    bg = (0, 0, 0)
    colormap = distinguishable_colormap(bg=bg)

    ren = window.Renderer()
    ren.background(bg)
    ren.background((0.7, 0.7, 0.7))
    ren.projection("parallel")

    actors = []
    texts = []
    for cluster, color, tracts in izip(streamlines, colormap, 
                                       glob.glob(sys.argv[1])) : # glob.glob(sys.argv[2])):
        if verbose :
            print color

        stream_actor = actor.line(cluster, [color]*len(cluster), linewidth=1)
        pretty_actor = auto_orient(stream_actor, ren.camera_direction(), 
                                   data_up=(0, 0, 1), show_bounds=True)
        pretty_actor_aabb = auto_orient(stream_actor, ren.camera_direction(), 
                                        bbox_type="AABB", show_bounds=True)
        actors.append(pretty_actor)

        s = tracts.split("/")
        print(s)
        text = actor.text_3d("Bundle i", font_size=60, 
                             justification="center", vertical_justification="top")        
        text = actor.text_3d(s[0], font_size=60, #s[1], font_size=60, 
                             justification="center", vertical_justification="top")
        texts.append(text)
        

#         if len(profile) != 0 :
#             s = profile.split("/")
#             print(profile)
#             p = np.load(profile)
#         #x = np.linspace(0, 1)
#         #p = np.sin(4 * np.pi * x) * np.exp(-5 * x)
#             fig = plt.figure(figsize=(800/300, 500/300), dpi=300)
#             ax = fig.add_subplot(111)
#             ax.set_title("FA tract profile")
#             ax.set_title(s[1])
#             ax.fill_between(np.arange(p[0].shape[0]), 
#                             p[0] - p[1], p[0] + p[1], 
#                             facecolor='y', alpha=0.5)
#                         #facecolor='0.75', alpha=1)

#             ax.plot(p[0], color='k', linewidth=5)
#         #ax.fill(p[0], 'r')
#             ax.grid(True)
#             color=(0.5,0.5,0.5)
#             ax.set_axis_bgcolor(color)  
#             arr = utils.matplotlib_figure_to_numpy(fig, dpi=300, transparent=True)
#             plt.close(fig)
#             figure_actor = actor.figure(arr, interpolation='cubic')
#             actors.append(figure_actor)
#             text = actor.text_3d('', font_size=32, 
#                                  justification="center", vertical_justification="top")
#             texts.append(text)

#         #ren.add(figure_actor)


    grid = actor.grid(actors, texts, cell_padding=(50, 100), cell_shape="rect")
    ren.add(grid)

    #ren.reset_camera_tight()
    show_m = window.ShowManager(ren, 
                                interactor_style=interactor.InteractorStyleBundlesGrid(actor))
    show_m.start()


if __name__ == "__main__":
    main()

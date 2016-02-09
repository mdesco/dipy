from __future__ import division

from itertools import izip

from dipy.viz import window, actor
from dipy.viz.colormap import distinguishable_colormap


from dipy.viz import interactor
from dipy.viz.utils import auto_orient

import nibabel as nib
import sys
import glob

verbose=False
def main():
    print(glob.glob(sys.argv[1]))
    
    streamlines = []
    for tracts in glob.glob(sys.argv[1]) :        
        if verbose :
            print(tracts)
        streamlines += [nib.streamlines.load(tracts).streamlines]

    bg = (0, 0, 0)
    colormap = distinguishable_colormap(bg=bg)

    ren = window.Renderer()
    ren.background(bg)
    ren.projection("parallel")

    actors = []
    texts = []
    for cluster, color, tracts in izip(streamlines, colormap, glob.glob(sys.argv[1])):
        if verbose :
            print color
        stream_actor = actor.line(cluster, [color]*len(cluster), linewidth=1)
        pretty_actor = auto_orient(stream_actor, ren.camera_direction(), 
                                   data_up=(0, 0, 1), show_bounds=True)
        pretty_actor_aabb = auto_orient(stream_actor, ren.camera_direction(), 
                                        bbox_type="AABB", show_bounds=True)

#        actors.append(stream_actor)
#        actors.append(pretty_actor_aabb)
        actors.append(pretty_actor)

#         text = actor.text_3d(str(len(cluster)), font_size=32, 
#                              justification="center", vertical_justification="top")
#         texts.append(text)

#         text = actor.text_3d("AABB", font_size=32, 
#                              justification="center", vertical_justification="top")
#         texts.append(text)

#         text = actor.text_3d("OBB", font_size=32, 
#                              justification="center", vertical_justification="top")
        text = actor.text_3d(tracts, font_size=32, 
                             justification="center", vertical_justification="top")
        texts.append(text)

    grid = actor.grid(actors, texts, cell_padding=(50, 100), cell_shape="rect")
    ren.add(grid)

    ren.reset_camera_tight()
    show_m = window.ShowManager(ren, 
                                interactor_style=interactor.InteractorStyleBundlesGrid(actor))
    show_m.start()


if __name__ == "__main__":
    main()

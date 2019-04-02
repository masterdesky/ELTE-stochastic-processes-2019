import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import random
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from datetime import datetime
import imageio

sns.set_style(style='whitegrid')
# Others
steps = 1
dpi = 80
fps = 60


def ANIMATE_VIDEO(path, video_path, gen_video_title, packages):
    
    video_title = video_path + gen_video_title
    
    nrows=1
    ncols=1
    figsize= 15
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize,figsize))

    axislabelsize = 20
    axisticksize = 17
    legendsize = 17

    # Measuring time for progress bar
    starttime = datetime.now()
    # Width of progress bar, shown on sys.stdout
    progressbar_width = 20
    
    ##ANIMATION STUFF BEGINS HERE##
    # Plot and save an image of the twobody system for time point i
    def animation(i):

        # The current positions of the spins
        Spins_positive_2D = Spins_2D[i][Spins_2D[i]>0]
        Spins_negative_2D = Spins_2D[i][Spins_2D[i]<0]

        axes.imshow(Spins_2D[i], cmap='Greys_r')

        # Axis labels
        axes.set_xlabel('Number of positive spins: {0}\nNumber of negative spins: {1}'.format(len(Spins_positive_2D), len(Spins_negative_2D)),
                        fontsize=axislabelsize)
        axes.tick_params(axis='both', which='major', labelsize=axisticksize)

        #
        #  CUSTOM COLORBAR
        #
        cmap = plt.cm.Greys_r  # define the colormap
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]

        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(-1, 1, 3)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # create a second axes for the colorbar
        ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

        #if sys.argv[1] == '1':
            # Don't show axes, only white background
            #axes.axis('off')

        plt.savefig(path + '_img{0:4d}.png'.format(i), dpi=dpi)    # Save next frame as png
        image = imageio.imread(path + '_img{0:4d}.png'.format(i)) # Load saved image
        writer.append_data(image)                                 # Append this image as the next frame to video

        # Clear the pyplot background for the next frame
        axes.cla()
        ax2.cla()

        # Delete the now useless image from frames' folder
        os.unlink(path + '_img{0:4d}.png'.format(i))

    with imageio.get_writer(video_title + '.mp4', fps=fps) as writer:
        for i in range(MIN_LIM, MAX_LIM):
            animation(i)
            sys.stdout.write("Progress: [{0}{1}] {2:.3f}%\tElapsed: {3}\tRemaining: {4}\r".format(u'\u2588' * int((i - MIN_LIM +1)/(MAX_LIM-MIN_LIM) * progressbar_width),
                                                                                                  u'\u2591' * (progressbar_width - int((i - MIN_LIM +1)/(MAX_LIM-MIN_LIM) * progressbar_width)),
                                                                                                  (i - MIN_LIM +1)/(MAX_LIM-MIN_LIM) * 100,
                                                                                                  datetime.now()-starttime,
                                                                                                  (datetime.now()-starttime)/(i - MIN_LIM +1) * ((MAX_LIM-MIN_LIM) - (i - MIN_LIM +1))))
            sys.stdout.flush()


# =============== MAIN ===============
Spins_2D = np.load('spinstate_2D.npy')

# Limits of animated parts
MIN_LIM = int(sys.argv[1])
MAX_LIM = int(sys.argv[2])

if MIN_LIM < 0:
    MIN_LIM = 0
if MAX_LIM > Spins_2D.shape[0]:
    MAX_LIM = Spins_2D.shape[0]
if MIN_LIM > MAX_LIM:
    MIN_LIM_TEMP = MIN_LIM
    MIN_LIM = MAX_LIM
    MAX_LIM = MIN_LIM_TEMP

ANIMATE_VIDEO(path = '.\\frames\\', video_path='.\\videos\\', gen_video_title=sys.argv[3], packages=20)
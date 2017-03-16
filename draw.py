import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os,sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def draw_bbox(subplot, im, bboxes,color='cyan',linewidth=2):
    #bbox: xmin, ymin, xmax, ymax
    #
    imshow(im)
    for b in bboxes:
        x=b[0]
        y=b[1]
        w=b[2]-x
        h=b[3]-y
        subplot.add_patch(patches.Rectangle(
            (x,y),   # (x,y)
            w,          # width
            h,          # height
        fill = 0,
        edgecolor=color,
        linewidth = linewidth
        ))

def fcn_color_map(im,fg,bg,out,LA=1,RA=0): #left alpha/right alpha
    out = out[...,np.newaxis]
    out = np.tile(out, (1,1,3))
    w = fg.shape[1]; h = fg.shape[0]
    min_max = lambda x: ((x-x.min())/(x.max()-x.min()))
    fg = min_max(fg)
    bg = min_max(bg)
    if w > h: f = figure(figsize=(16,27))
    else: f = figure(figsize=(9,32))
    #----------------------#
    f.add_subplot(521);imshow(fg);title('FG')
    f.add_subplot(522);imshow(im,alpha=LA);imshow(fg,alpha=RA)
    #----------------------#
    f.add_subplot(523);imshow(bg);title('BG')
    f.add_subplot(524);imshow(im,alpha=LA);imshow(bg,alpha=RA)
	#----------------------#
    f.add_subplot(525);imshow(fg-bg);title('FG-BG')
    f.add_subplot(526);imshow(im,alpha=LA);imshow(fg-bg,alpha=RA)
    #----------------------#
    f.add_subplot(527);imshow(out);title('ArgMax')
    f.add_subplot(528);imshow(im,alpha=LA);imshow(out,alpha=RA)
	#----------------------#
    f.add_subplot(529);imshow(fg+bg);title('FG+BG')
    f.add_subplot(5,2,10);imshow(im,alpha=LA);imshow(fg+bg,alpha=RA)


def fcn_3d(fg,figsize=(9,9)):
    fig = figure(figsize=figsize)
    X,Y = np.meshgrid(np.linspace(0,fg.shape[1],fg.shape[1]),np.linspace(0,fg.shape[0],fg.shape[0]))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, fg, cmap=cm.hot, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

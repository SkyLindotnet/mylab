import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os,sys
def save_dict(name,dictionary):
    np.save(name, dictionary) 
def load_dict(name):
    return np.load(name).item()
def draw_bbox(subplot, im, bboxes,color='cyan',linewidth=2):
    #bbox: xmin, ymin, xmax, ymax
    
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

def get_fddb_fcn(s):
   #INPUT
    # TYPE1 : ??/....??/??/??/2002_08_02_big_img_198.jpg
    # TYPE2 : ??/??/....??/??/??/2002/08/02/big/img_198.jpg
   #OUTPUT
    # fg,bg,out(i.e. argmax)

    fcn_out_dir = '/home/rick/Space/work/FDDB/FCN_OUT'
    s = s.split('.')[0].replace('\n','').replace(' ','')

    if len(s.split('_')) > 3: s = s.split('/')[-1]
    else: s = '_'.join(s.split('/')[-5:])

    target = os.path.join(fcn_out_dir,s+'.npy')
    assert os.path.exists(target), "[!] {} not found".format(target.replace('.npy',' .npy'))
    D = load_dict(target)
    return D['fg'], D['bg'], D['out'], 

def get_fddb_list():
	with open('/home/rick/Space/work/FDDB/data/Annotations/FDDB_all.txt','r') as f:
		l = f.readlines()
	return [ll.replace("\n",'') for ll in l]

def get_fddb_img(num=1, imgread=True,usePIL=True):
    pfile = '/home/rick/Space/work/FDDB/data/Annotations/FDDB_all.txt'
    with open(pfile,'r') as f:
        img_path = f.readlines()
    
    img_path = [x.replace('\n','') for x in img_path]
    num_img = len(img_path);
    assert num<=num_img, "too many required images"
    idx = npr.choice(num_img,num,replace=0)
    img_path = np.array(img_path)
    imgs = img_path[idx]
    for i in imgs:
        print "Image: {}".format(i)
    if not imgread:
        imgs = imgs
    else:
        if usePIL:
            imgs = [Image.open(x) for x in imgs]
        else:
            imgs = [imread(x) for x in imgs]
    if(num == 1):
        return imgs[0]
    else:
        return imgs
    
def get_wider_img(num=1, imgread=True,usePIL=True, setname='all'):
    #set: all/train/test/val/
    
    allfile = '/home/rick/Documents/Models/WIDER_FACE/unzips/imgpath/all.txt'
    trainfile ='/home/rick/Documents/Models/WIDER_FACE/unzips/imgpath/train.txt'
    testfile='/home/rick/Documents/Models/WIDER_FACE/unzips/imgpath/test.txt'
    valfile='/home/rick/Documents/Models/WIDER_FACE/unzips/imgpath/val.txt'
    
    files = [allfile,trainfile,testfile,valfile]
    names = ['all','train','test','val']
    idx = 0
    if setname in names:
        idx = names.index(setname)
    
    
    with open(files[idx],'r') as f:
        img_path = f.readlines()
    
    img_path = [x.replace('\n','') for x in img_path]
    num_img = len(img_path);
    assert num<=num_img, "too many required images"
    idx = npr.choice(num_img,num,replace=0)
    img_path = np.array(img_path)
    imgs = img_path[idx]
    for i in imgs:
        print "Image: {}".format(i)
    if not imgread:
        imgs = imgs
    else:
        if usePIL:
            imgs = [Image.open(x) for x in imgs]
        else:
            imgs = [imread(x) for x in imgs]
    if(num == 1):
        return imgs[0]
    else:
        return imgs



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def fcn_3d(fg):
    fig = figure()
    X,Y = np.meshgrid(np.linspace(0,fg.shape[1],fg.shape[1]),np.linspace(0,fg.shape[0],fg.shape[0]))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, fg, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os,sys


def get_wider_img(num=1, imgread=True,usePIL=True, setname='all'):
    #set: all/train/test/val/
    '''

    :param num:
        Number of images required:
            if give 1: return a single image
            if > 1 : return a list of image
    :param imgread:
        Read the image or just return the image path
        if imgread: return an opened image
        if not imgread: return a image path
    :param usePIL:
        The way to open image:
            if usePIL: read image using PIL.Image.open
            if not usePIL: read image using pylab.imread
    :param setname:
        From which set of wider set to get images:
            in ['all','train','test','val']
    :return:
        Either opened image(s) or just image path(s)
    '''
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


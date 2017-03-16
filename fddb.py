import numpy as np
from pylab import *
import matplotlib.patches as patches
import numpy.random as npr
from PIL import Image
import os,sys
from .io import *
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
    num_img = len(img_path)
    assert num<=num_img, "too many required images"
    idx = npr.choice(num_img,num, replace=False)
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

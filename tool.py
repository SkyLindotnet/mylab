import numpy as np
from easydict import EasyDict as edict
def save_dict(name,dictionary):
    '''

    :param name: full path to save the dict (suggested extension: npy)
    :param dictionary: a dict variable
    :return:
    '''
    np.save(name, dictionary) 
def load_dict(name):
    '''

    :param name: full path to a saved dict
    :return: loaded dict
    '''
    return np.load(name).item()


def parse_file(f, full_path=False):
    '''
    
    :param f: file_path
    :param full_path: set True to disable path prune
    :return: 
    1. ordered key of dict
        
    2. dict, key is $1 returned, value is also a dict
    in each dict, it has keys:
        'name': path (effected by arg 'full_path');
        'num' : number of faces
        'coords': a list of string
        
    '''
    def deal_file_path(s,full_path):
        if full_path:
            return s
        else:
            s = s.strip('.jpg')
            return '/'.join(s.split('/')[-5:])

    with open(f,'r') as f:
        lines = f.readlines()
        lines = [l.strip('\n') for l in lines]

    at = 0
    ret = edict()
    keys = []
    while(at < len(lines)-1):
        d = edict()
        this = lines[at]
        name = deal_file_path(this, full_path)
        keys.append(name)
        at += 1

        this = int(lines[at])
        d['num']  = this; at += 1


        coords = []
        for _ in range(this):
            coords.append(lines[at]); at += 1

        d['coords'] = coords

        ret[name] = d

    return keys, ret


def write_out(keys, d, path):
    s = []
    for k in keys:
        s.append(k)
        d_i = d[k]
        s.append(str(d_i['num']))
        s.extend(d_i['coords'])
    s = '\n'.join(s) + '\n'

    with open(path,'w') as f:
        f.write(s)


import numpy as np

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


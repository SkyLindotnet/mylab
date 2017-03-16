import numpy as np

def save_dict(name,dictionary):
    np.save(name, dictionary) 
def load_dict(name):
    return np.load(name).item()


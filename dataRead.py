'''
Date: 04-04-2018
Author: He Zhao
Institute: York University, Vision Lab
Introduction : Reading mat file and preprocessing

'''
from __future__ import division
import os, sys
import scipy.io as io
import numpy as np

def readMat(data_path):
    
    return io.loadmat(data_path)['result']

def normalize(data):
    ''' 
        convert numpy array into normalized version based on each channel;
        if input is list, convert to arrary first
    '''
    # import pdb; pdb.set_trace()

    if type(data) == list:
        data = np.expand_dims(np.asarray(data), 0)

    dims, length = data.shape

    for i in range(dims):
		data[i,:] = data[i,:] / np.max(data[i,:])

    return data 

if __name__ == '__main__':
    data_path = ''    
    readMat(data_path)


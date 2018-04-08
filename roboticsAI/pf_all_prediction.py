'''
Author: He Zhao
Introduction: Particle Filter Implementation for belief propagation research
Insitute: York University, Vision Lab, supervised by Richard Wildes
Version: pure prediction
TODO: observation function
TODO: visualization
'''

from __future__ import division
from math import *
import random, os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from dataRead import readMat, normalize
from config import data_config
from utils import motionFCN, gradient_1, gradient_2, resample2, resample1
import numpy as np
import matplotlib.pyplot as plt
from customPF import particles

if __name__ == '__main__':

    # prepare data & choose arbitary dimension
    data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
    measurements = normalize(readMat(data_path))
    print(measurements.shape)
    Dims, Length = measurements.shape

    # dim = 7
    dim = int(random.random() * Dims)
    data_mean = np.mean(measurements[dim, :])
    
    # init 100 particles 
    N = 50
    P = []
    Estimation = []

    for i in range(N):
        x = particles()
        # x.set(random.uniform(0, 1), 0.01, 0.0001, 1)
        x.set(random.gauss(data_mean, 0.2), 0.1, 0.001, 1)
        P.append(x)

    # gradient
    g_1 = gradient_1(measurements[dim, :])
    g_2 = gradient_2(measurements[dim, :])
    
    # run motion model
    for t in range(Length):
        measurement = measurements[dim, t]

        for n in range(N):

            P[n].motion(motionFCN, v=g_1[t], a=g_2[t], t=1)
            # P[n].motion(motionFCN, v=g_1[t], a=0, t=1)
        
        # evaluation and store
        Estimation.append(np.mean([y.x for y in P]))

    # plot result
    plt.figure(1)
    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(measurements[dim, :])
    plt.plot(Estimation)
    plt.legend(['True data', 'Estimation'], loc='upper left')
    plt.show()






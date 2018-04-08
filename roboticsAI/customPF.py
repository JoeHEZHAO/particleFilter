'''
Author: He Zhao
Introduction: Particle Filter Implementation for belief propagation research
Insitute: York University, Vision Lab, supervised by Richard Wildes
TODO: observation function
TODO: visualization particle distribution
TODO: Further improve resampling methods
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

class particles:

    def __init__(self):
        self.weight = 0.0
        self.x = 0.0
        self.forward_noise = 0.0
        self.sense_noise = 0.0
        self.boundary = 0.0

    def set(self, x, forward_noise, sense_noise, boundary):
        self.x = x
        self.forward_noise = forward_noise
        self.sense_noise = sense_noise
        self.boundary = boundary

    def motion(self, motion, v=1, a=0, t=1):

        self.x = motion(self.x, v, a, t) + random.gauss(0.0, self.forward_noise)

        ''' out of boundary would be taken care of by resampling
        if self.x >= 1.0:
            self.x = 0.9
        elif self.x <= 0.0:
            self.x = 0.1

        self.x %= self.boundary
        '''

    def Gaussian(self, mu, sigma, x):
        '''
        Compute the gaussian prob between two variable
        '''
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))

    def measurement_prob(self, measurement):

        dist = self.x

        prob = self.Gaussian(dist, self.sense_noise, measurement)

        return prob

    def __repr(self):
        return '[x=%.3s, weight=%.3s]' % (str(self.x,), str(self.weight))

def eval(r, p, N):
    sum = 0.0
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r) % N
        err = sqrt(dx * dx)
        sum += err
    return sum / float(len(p))

if __name__ == '__main__':

    # prepare data & choose arbitary dimension
    data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
    measurements = normalize(readMat(data_path))
    print(measurements.shape)
    Dims, Length = measurements.shape

    # dim = 7
    dim = int(random.random() * Dims)
    data_mean = np.mean(measurements[dim, :])

    # highlight
    pred_range = [64, 74]
    
    # init 100 particles 
    N = 50
    P = []
    Estimation = []

    for i in range(N):
        x = particles()
        # x.set(random.uniform(0.5, 1), 0.005, 100.0, 1)
        x.set(random.gauss(data_mean, 0.1), 0.005, 5.0, 1)
        P.append(x)

    # gradient
    g_1 = gradient_1(measurements[dim, :])
    g_2 = gradient_2(measurements[dim, :])
    
    # run motion model
    for t in range(Length):
        measurement = measurements[dim, t]

        if pred_range[0] < t < pred_range[1]:

            for n in range(N):
                P[n].motion(motionFCN, v=g_1[t], a=g_2[t], t=1)
                # P[n].motion(motionFCN, v=g_1[t], a=0, t=1)

        else:

            # assign new Particle and Weight matrix
            W = []

            # motion
            for n in range(N): 

                P[n].motion(motionFCN, v=g_1[t], a=g_2[t], t=1)

                W.append(P[n].measurement_prob(measurement)) 

            ''' normal weights turns tobe important '''
            W1 = [i / (sum(W) + 1e-10) for i in W]
            W = W1

            '''plot weight changing'''
            # print(sum(W))
            # plt.plot(W)
            # plt.show()

            ''' resample1 from robotAI course '''
            # P = resample1(W, P)

            ''' resample2 '''
            P = resample2(W, P)
        
        # plot Particles after either pred or pred+update
        # plt.plot([y.x for y in P])
        # plt.show()
        
        # evaluation and store
        score = np.mean([y.x for y in P])
        Estimation.append(score)

    # plot result
    plt.figure(1)
    plt.gca().set_color_cycle(['red', 'green'])
    plt.plot(measurements[dim, :])
    plt.plot(Estimation)
    plt.legend(['True data', 'Estimation'], loc='upper left')
    plt.show()






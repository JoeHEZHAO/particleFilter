'''
Author: He Zhao
Institute: Vision Lab, York University
Date: 04-10-2018
Introduction: Build up Propagation class and test

'''
from __future__ import division
import os, sys, cv2, random, copy
import numpy as np
from math import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from ParticleFilter import particles
from dataRead import readMat, normalize
from config import data_config
from utils import motionFCN, gradient_1, gradient_2, resample2, resample1
from matplotlib import pyplot as plt


class propagation(object):

    def __init__(self, particles, motionfcn, measurements, gradient_1, gradient_2):

        self.estimation = []
        self.motionfcn= motionfcn
        self.prediction = []
        self.particles = particles
        self.measurements = measurements
        self.gradient_1 = gradient_1
        self.gradient_2 = gradient_2
        self.particle_num = len(self.particles)
        self.data_length = len(self.measurements)

        if self.gradient_2 is None:
            self.gradient_2 = np.zeros(self.data_length)
            print("using motion model with velocity only")
        else:
            print("using motion model with velocity and acceleration")

    def pred(self, t_idx):

        for n in range(self.particle_num):

            self.particles[n].motion(self.motionfcn, v=self.gradient_1[t_idx], a=self.gradient_2[t_idx], t=1)

        self.prediction.append(np.mean([y.x for y in self.particles]))

    def measurement_update(self, t_idx):

        W = []

        for n in range(self.particle_num):

            W.append(self.particles[n].measurement_prob(self.measurements[t_idx]))

        W1 = [i / (sum(W) + 1e-10) for i in W]
        W = W1

        self.particles = resample2(W, self.particles)

        self.estimation.append(np.mean([y.x for y in self.particles]))


if __name__ == '__main__':

    # prepare data & choose arbitary dimension
    data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
    data = normalize(readMat(data_path))
    # print(data.shape)
    Dims, Length = data.shape

    # dim = 7
    dim = int(random.random() * Dims)
    data_mean = np.mean(data[dim, :])

    # highlight
    pred_range = [64, 74]

    # init 100 particles
    N = 100
    P = []
    Estimation = []

    for i in range(N):
        x = particles()
        # x.set(random.uniform(0.5, 1), 0.005, 100.0, 1)
        x.set(random.gauss(data_mean, 0.2), 0.005, 5.0, 1)
        P.append(x)

    g_1 = gradient_1(data[dim, :])
    g_2 = gradient_2(data[dim, :])


    # motion with both velocity and acceleration
    P1 = copy.deepcopy(P)
    prog = propagation(P1, motionFCN, data[dim, :], g_1, g_2)

    for t_idx in range(len(data[dim,:])):

        prog.pred(t_idx)

        prog.measurement_update(t_idx)


    # motion with only velocity
    P2 = copy.deepcopy(P)
    prog1 = propagation(P2, motionFCN, data[dim, :], g_1, None)

    for t_idx in range(len(data[dim, :])):

        prog1.pred(t_idx)

        prog1.measurement_update(t_idx)

    plt.figure(1)
    plt.gca().set_color_cycle(['red', 'green', 'blue'])
    plt.plot(data[dim, :])
    plt.plot(prog.estimation)
    plt.plot(prog1.estimation)
    plt.legend(['velocity with acceleration', 'velocity only', 'True data'], loc='upper left')
    plt.show()

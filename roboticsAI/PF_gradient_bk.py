'''
Author: He Zhao
Institute: Vision Lab, York University
Date: 04-16-2018
Introduction: Use back-ward gradient to propagate distribution

If propagation results should not be too ideal, otherwise it means however change, it can still predict right distribution
Thus, there is no meaning of anomaly detection

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
from Propagation import propagation

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
    x.set(random.gauss(data_mean, 0.2), 0.005, 5.0, 1)
    P.append(x)

g_1 = gradient_1_bk(data[dim, :])
g_1_bk = gradient_1(data[dim, :])
g_2 = gradient_2(data[dim, :])
g_2_bk = gradient_2_bk(data[dim, :])

# motion with both velocity and acceleration
P1 = copy.deepcopy(P)
prog = propagation(P1, motionFCN, data[dim, :], g_1, g_2)

for t_idx in range(len(data[dim,:])):

    prog.pred(t_idx)

    prog.measurement_update(t_idx)

# motion with only velocity
P2 = copy.deepcopy(P)
prog1 = propagation(P2, motionFCN, data[dim, :], g_2, g_2_bk)

for t_idx in range(len(data[dim, :])):

    prog1.pred(t_idx)

    prog1.measurement_update(t_idx)

plt.figure(1)
plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.plot(data[dim, :])
plt.plot(prog.estimation)
plt.plot(prog1.estimation)
plt.title("Gradient is using back-wards direction")
plt.legend(['True data', 'velocity with backwards', 'velocity with center'], loc='upper left')
plt.show()

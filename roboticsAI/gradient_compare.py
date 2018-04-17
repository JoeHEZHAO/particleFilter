'''
Date: 04-16-2018
Introduction: compute and display different gradient methods


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

g_2 = gradient_2(data[dim, :])
g_1 = gradient_1(data[dim, :])

plt.gca().set_color_cycle(["red", "green", "yellow"])
# plt.plot(g_1_bk)
plt.plot(g_2)
# plt.plot(g_1)
plt.legend(["G1 with bk", "G2", "G1 with center"] , loc="upper left")
plt.show()

import os, sys, cv2, random, copy
import numpy as np
from ParticleFilter import particles
from Propagation import propagation
from utils import motionFCN, gradient_1, gradient_2, resample2, update_acceleration
from math import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from dataRead import readMat, normalize
from config import data_config, covariance_matrix
from matplotlib import pyplot as plt

# prepare data & choose arbitary dimension
data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
data = normalize(readMat(data_path))
Dims, Length = data.shape

# Convert acceleration to covariance and weighted
g_1 = []
g_2 = []

# init gradient for all data, all channel
for k in range(8):

    g_1.append(gradient_1(data[k, :]))
    g_2.append(gradient_2(data[k, :]))

# convert list to array
g_1 = np.asarray(g_1)
g_2 = np.asarray(g_2)
g_2_update = copy.deepcopy(g_2)

# update gradient
for k in range(Length):

    g_2_update[:,k] = update_acceleration(g_2[:, k], covariance_matrix)

print("FInish updateing acceleration")

print("Total Dimension is {}".format(Dims))

# choose random dim from [0, 7] for testing random directions
dim = random.randint(0, 7)
data_mean = np.mean(data[dim, :])
print("Random choose dimension {}".format(dim))

# init 100 particles
N = 100
P = []

for i in range(N):
    x = particles()
    x.set(random.gauss(data_mean, 0.2), 0.005, 5.0, 1)
    P.append(x)

# propagating with updated 2ed gradient
P1 = copy.deepcopy(P)
prog = propagation(P1, motionFCN, data[dim, :], g_1[dim, :], g_2_update[dim, :])

for t_idx in range(Length):

    prog.pred(t_idx)
    prog.measurement_update(t_idx)

# propagation with original 2ed gradient
P2 = copy.deepcopy(P)
prog2 = propagation(P2, motionFCN, data[dim, :], g_1[dim, :], g_2[dim, :])

for t_idx in range(Length):

    prog2.pred(t_idx)
    prog2.measurement_update(t_idx)

plt.figure(1)
plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.plot(data[dim, :])
plt.plot(prog.estimation)
plt.plot(prog2.estimation)
plt.legend(['True data', 'updated acceleration', 'original acceleration'], loc='upper left')
plt.show()

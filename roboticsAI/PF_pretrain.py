'''
Author: He Zhao
Institute: Vision Lab, York University
Date: 04-16-2018
Introduction: Use pre-defined frames as training for particle init
              Gradually take-in new frames to re-init particles, as a way to mingle all new events
              Do not normalize data;
              Current result is pretty bad since particle value varies too much

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
from utils import motionFCN, gradient_1, gradient_2, resample2, resample1, gradient_1_bk, gradient_2_bk, average_gradient
from matplotlib import pyplot as plt
from Propagation import propagation

# prepare data & choose arbitary dimension
data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
data = readMat(data_path)
# data = normalize(data)

# if data is canoe but not standard
data = data.transpose()
Dims, Length = data.shape
print(Dims, Length)

'''
    use first several frames to generate the particles
    assume does not know the future frames
'''
training_stage = [0, 50]

# dim = 7
dim = int(random.random() * Dims)
data_mean = np.mean(data[dim, :])
print("The data mean from all data is {}".format(data_mean))
data_mean = np.mean(data[dim, training_stage[0]:training_stage[1]])
print("The data mean from only training data is {}".format(data_mean))
data_variance = np.max(data[dim, :]) - np.min(data[dim, :])
print("The total data variance is {}".format(data_variance))
data_variance = np.max(data[dim, training_stage[0]:training_stage[1]]) - np.min(data[dim, training_stage[0]:training_stage[1]])
print("The traingin stage variance is {}".format(data_variance))

# init 100 particles
N = 100
P = []
Estimation = []

for i in range(N):
    x = particles()
    x.set(random.gauss(data_mean, data_variance*5), 0.5, data_variance**2, 1)
    P.append(x)

g_1_bk = gradient_1_bk(data[dim, :])
g_1_bk = average_gradient(g_1_bk, 3)

''' particles init value and gradient is fine '''
# plt.plot(g_1_bk)
# plt.plot([y.x for y in P])
# plt.show()

g_1= gradient_1(data[dim, :])
g_2 = gradient_2(data[dim, :])
g_2_bk = gradient_2_bk(data[dim, :])
g_1_random = [random.gauss(0.0, 0.1) * .5 for x in range(Length)]
print(np.min(g_1_random), np.max(g_1_random))
print(np.min(g_1_bk), np.max(g_1_bk))

'''  motion with both velocity and acceleration '''
P1 = copy.deepcopy(P)
# prog = propagation(P1, motionFCN, data[dim, :], g_1_bk, g_2_bk)
prog = propagation(P1, motionFCN, data[dim, :], g_1_bk, None)

for t_idx in range(len(data[dim,:])):

    prog.pred(t_idx)

    prog.measurement_update(t_idx)

    ''' plot particle value and weights to debug '''
    # plt.plot([y.x for y in P1])
    # plt.plot([y.weight for y in prog.particles])
    # plt.show()

''' motion with only velocity '''
P2 = copy.deepcopy(P)
# prog1 = propagation(P2, motionFCN, data[dim, :], g_1, g_2)
# prog1 = propagation(P2, motionFCN, data[dim, :], g_1, None)
prog1 = propagation(P2, motionFCN, data[dim, :], g_1, None)

for t_idx in range(len(data[dim, :])):

    prog1.pred(t_idx)

    if 500 < t_idx < 1000:
        prog1.estimation.append(prog1.prediction[t_idx])
    else:
        prog1.measurement_update(t_idx)

plt.figure(1)
plt.gca().set_color_cycle(['red', 'green', 'blue'])
plt.plot(data[dim, :])
plt.plot(prog.estimation)
plt.plot(prog1.estimation)
plt.title("Gradient is using back-wards direction")
plt.legend(['True data', 'velocity with backwards', 'velocity center'], loc='upper left')
plt.show()

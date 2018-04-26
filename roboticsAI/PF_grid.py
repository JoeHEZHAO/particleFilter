'''
Author: He Zhao
Institute: Vision Lab, York University
Date: 04-22-2018
Introduction: split soe feature into grids and compute the results
'''

from __future__ import division
import os, sys, cv2, random, copy, pims
import numpy as np
from math import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from videofig_class import videofig
from ParticleFilter import particles
from dataRead import readMat, normalize
from config import data_config
from utils import motionFCN, gradient_1, gradient_2, resample2, resample1, gradient_1_bk, gradient_2_bk
from matplotlib import pyplot as plt
from Propagation import propagation
from grid_video import redraw_fn, customVideo
import argparse

''' parse args'''
parser = argparse.ArgumentParser(description='choose grid')
parser.add_argument('grid', type=int, help='input 0 - 100 grd number')

args = parser.parse_args()

''' prepare data '''
data_config['data_name'] = 'Canoe_grid.mat'
data_path = os.path.join(data_config['root_dir'], data_config['data_name'])
data = readMat(data_path)
grids, Length, channels = data.shape
print(grids, Length, channels)

''' readin chosen grid '''
grid = args.grid
print("Choosing grid {}".format(grid))
# print("Choosing dim {}".format(dim))

result = []
for dim in range(8):

    print("Working on feature channel {}".format(dim + 1))

    ''' squeeze and transpose data '''
    data1 = data[grid, :, dim+1]
    data1 = data1.transpose()

    ''' split train and test data '''
    data_train = data1[:50]

    ''' compute mean and variance '''
    data_mean = np.mean(data_train)
    data_variance = np.max(data_train) - np.min(data_train)
    print("training stage data max is {}".format(np.max(data_train)))
    print("training stage data min is {}".format(np.min(data_train)))
    print("traing stage data mean is {}".format(data_mean))
    print("traing stage data variance is {}".format(data_variance))

    ''' init particles '''
    N = 100
    P = []

    for i in range(N):
        x = particles()
        # x.set(random.uniform(data_mean - data_variance, data_mean + data_variance), .0001, 5.0, 1)
        # x.set(random.gauss(data_mean, .2), .0001, 5.0, 1)
        x.set(random.uniform(data_mean - data_variance, data_mean + data_variance), .001, data_variance**2, 1)
        # x.set(random.gauss(data_mean, data_variance / 5), .001, data_variance**2, 1)
        P.append(x)

    ''' compute gradient '''
    g_1 = gradient_1_bk(data1)
    print("gradient range is {} to {}".format(np.max(g_1), np.min(g_1)))

    ''' Propagate '''
    P1 = copy.deepcopy(P)
    prog = propagation(P1, motionFCN, data1, g_1, None)

    for t_idx in range(Length):

        prog.pred(t_idx)

        if t_idx < 0:
            prog.measurement_update(t_idx)
        else:
            prog.estimation.append(prog.prediction[t_idx])
            # prog.measurement_update(t_idx)

    ''' result is stored in clock-wise direction '''
    result.append(prog.estimation)

''' define axis range for display '''
axis_range = []
for i in range(8):
    value_range = []
    value_range.append(np.min(result[i]) - np.min(result[i]) / 2)
    value_range.append(np.max(result[i]) + np.max(result[i]) / 2)
    axis_range.append(value_range)

''' read video for display '''
video_root = '/home/zhufl/Data/anomalous/Canoe/'
video_name = 'input.avi'
video_path = os.path.join(video_root, video_name)
# print(vid._shape)
vid = pims.Video(video_path)
vid_h, vid_w, vid_c = vid._shape

''' set up videofid and redraw_fn for display '''
xdata = np.arange(0.0, Length, 1.0)
def redraw_fn(f, arg):
    img = vid[f]
    if not redraw_fn.initialized:

        '''  set video '''
        redraw_fn.im = arg[0].imshow(img, animated=True)
        arg[0].set_title("Grid {}".format(grid))

        redraw_fn.lines = []
        for i in range(8):

            ''' setup primary lines for true data and prediction '''
            line = arg[i + 1]
            line.l1, = line.plot([], [], lw=2, color='blue')
            line.l2, = line.plot([], [], lw=2, color='red')

            ''' buidup secondary axis for error '''
            line.twin = line.twinx()
            err = np.absolute(np.max(data[grid, :, (8-i)%8+1] - result[(8-i)%8][:]))
            line.twin.set_ylim(0, err * 3)
            line.twin.l3, = line.twin.plot([], [], lw=2, color='green')

            redraw_fn.lines.append(line)

        redraw_fn.initialized = True

    else:
        '''
        Reminder that soe directions are clock-wise in response data in format [0:8, :]
        '''
        redraw_fn.im.set_array(img)

        for idx, line in enumerate(redraw_fn.lines):
            line.l1.set_data(xdata[:f], result[(8-idx)%8][:f])
            line.l2.set_data(xdata[:f], data[grid, :f, (8-idx)%8 + 1])
            line.twin.l3.set_data(xdata[:f], np.absolute(data[grid, :f, (8-idx)%8 + 1] - result[(8-idx)%8][:f]))

redraw_fn.initialized = False
video = customVideo(len(vid), redraw_fn, vid_h, vid_w, axis_range, grid=True, play_fps=50)
video.run()

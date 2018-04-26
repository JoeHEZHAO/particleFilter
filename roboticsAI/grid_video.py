'''
Author: He Zhao
Institute: Vision Lab, York University
Date: 04-24-2018
Introduction: Define videofig and redraw_fn for grid display
'''

from __future__ import division
import os, sys, cv2, random, copy, pims
import numpy as np
from math import *
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
from videofig_class import videofig
from matplotlib import pyplot as plt
import matplotlib.ticker as pltticker
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

''' set videofig class for display '''
class customVideo(videofig):

    def __init__(self, num_frames, redraw_func, vid_h, vid_w, axis_range, grid=False, play_fps=25, big_scroll=30, key_func=None, *args):

        # init variables
        self.num_frames = num_frames
        self.redraw_func = redraw_func
        self.play_fps = play_fps
        self.big_scroll = big_scroll
        self.key_func = key_func
        self.grid = grid
        self.vid_h = vid_h
        self.vid_w = vid_w
        self.axis_range = axis_range

        # Check arguments
        self.check_int_scalar(self.num_frames, 'num_frames')
        self.check_callback(self.redraw_func, 'redraw_func')
        self.check_int_scalar(self.play_fps, 'play_fps')
        self.check_int_scalar(self.big_scroll, 'big_scroll')
        if self.key_func:
            self.check_callback(self.key_func, 'key_func')

        # Set initial player state
        self.play_running = False
        self.play_anim = None

        # Initialize figure
        self.fig_handle = plt.figure()

        # init axes
        self.axes_handle_init()

        # init scrollvar handler
        self.scroll_handle_init()

    # override axes function
    def axes_handle_init(self):
        self.axes_list = []

        # display video
        self.axes_handle = plt.axes([0, 0.50, 0.2, 0.2])

        if self.grid:

            ''' define grid interval and color '''
            dx = self.vid_w / 10
            dy = self.vid_h / 10
            locx = pltticker.MultipleLocator(base=dx)
            locy = pltticker.MultipleLocator(base=dy)

            ''' set grid '''
            self.axes_handle.xaxis.set_major_locator(locx)
            self.axes_handle.yaxis.set_major_locator(locy)
            self.axes_handle.grid(which='major', axis='both', linestyle='-')

            ''' set xlim, ylim and draw grid number '''
            self.axes_handle.set_xlim(0, self.vid_w)
            self.axes_handle.set_ylim(self.vid_h, 0)

            ''' Find number of gridsquares in x and y direction '''
            nx=abs(int(float(self.axes_handle.get_xlim()[1]-self.axes_handle.get_xlim()[0])/float(dx)))
            ny=abs(int(float(self.axes_handle.get_ylim()[1]-self.axes_handle.get_ylim()[0])/float(dy)))

            ''' Add some labels to the gridsquares '''
            for j in range(ny):
                y=dy/2+j*dy
                for i in range(nx):
                    x=dx/2.+float(i)*dx
                    self.axes_handle.text(x,y,'{:d}'.format(i+j*nx),color='w',ha='center',va='center')

        self.axes_list.append(self.axes_handle)

        ''' display eight soe directions '''
        col = 0
        row = 0

        data_legend = ['right', 'right-up', 'up', 'left-up', 'left', 'left-bottom', 'bottom', 'right-bottom']
        data_range =  np.zeros((8, 2), np.float)

        for i in range(8):

             self.axes_list.append(plt.axes([0.2 + row%2*0.4, 0.9 - (col+1)*0.2, 0.35, 0.175]))

             if i == 3:
                 col = 0
                 row = 1
             else:
                 col += 1

        for i in range(1, 9):
            self.axes_list[i].set_ylim(self.axis_range[(8-i+1)%8][0], self.axis_range[(8-i+1)%8][1])
            self.axes_list[i].set_xlim(0.0, self.num_frames)
            self.axes_list[i].set_title(data_legend[i-1])
            self.axes_list[i].grid()

        return self.axes_list

    def scroll_handle_init(self):

        '''  Build scrollbar '''
        self.scroll_axes_handle = plt.axes([0, 0, 1, 0.015], facecolor='lightgoldenrodyellow')
        self.scroll_handle = Slider(self.scroll_axes_handle, '', 0.0, self.num_frames - 1, valinit=0.0)

length = 120
xdata = np.arange(0.0, length, 1.0)

def redraw_fn(f, arg):
    img = vid[f]

    if not redraw_fn.initialized:
        # first display video
        redraw_fn.im = arg[0].imshow(img, animated=True)
        redraw_fn.l1, = arg[1].plot([], [], lw=2, color='blue')
        redraw_fn.l2, = arg[2].plot([], [], lw=2, color='blue')
        redraw_fn.l3, = arg[3].plot([], [], lw=2, color='blue')
        redraw_fn.l4, = arg[4].plot([], [], lw=2, color='blue')
        redraw_fn.l5, = arg[5].plot([], [], lw=2, color='blue')
        redraw_fn.l6, = arg[6].plot([], [], lw=2, color='blue')
        redraw_fn.l7, = arg[7].plot([], [], lw=2, color='blue')
        redraw_fn.l8, = arg[8].plot([], [], lw=2, color='blue')
        redraw_fn.initialized = True
    else:
        '''
        Reminder that soe directions are clock-wise in response data in format [0:8, :]
        '''
        redraw_fn.im.set_array(img)
        redraw_fn.l1.set_data(xdata[0 : f],data[0, : f])
        redraw_fn.l2.set_data(xdata[0 : f],data[8-1, : f])
        redraw_fn.l3.set_data(xdata[0 : f],data[8-2, : f])
        redraw_fn.l4.set_data(xdata[0 : f],data[8-3, : f])
        redraw_fn.l5.set_data(xdata[0 : f],data[8-4, : f])
        redraw_fn.l6.set_data(xdata[0 : f],data[8-5, : f])
        redraw_fn.l7.set_data(xdata[0 : f],data[8-6, : f])
        redraw_fn.l8.set_data(xdata[0 : f],data[8-7, : f])

if __name__ == '__main__':
    pass

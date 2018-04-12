'''
Design motion model for Particle Filter

'''

import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
from config import covariance_matrix
from math import *
import random
import numpy as np

def motionFCN(x, v=0, a=0, t=1):
    '''
    pos = pos + velocity*time  + 0.5*a*t^2
    '''

    return x + v*t + 0.5*a*t**2

def gradient_1(data):
    return np.gradient(data)

def gradient_2(data):
    return np.gradient(data, 2)

def resample2(W, P):
        w_len = len(W)
        indices = []
        P_new = []

		# cumsum
        C = [0.] + [sum(W[:i+1]) for i in range(w_len)]
        u0, j = np.random.rand(), 0

		# generate rising order random value to decide if discard weights
        b = [(u0+i)/w_len for i in range(w_len)]

        for u in b:

            while j < w_len and u > C[j]:
				j+=1
            indices.append(j-1)

        for i in range(len(W)):
            P_new.append(P[indices[i]])

        return P_new

def resample1(W, P):
    New_P = []
    N = len(W)

    index = int(random.random() * N)
    beta = 0.0
    mw = max(W)
    for i in range(N):
        beta += 2.0 * mw
        while beta > W[index]:
            beta -= W[index]
            index = (index + 1) % N
        New_P.append(P[index])

    P = New_P

    return P

def update_acceleration(gradient, covariance_matrix):

    '''
    Input: [8,] second order gradient matrix, for only one time step;
           [8, 8] covariance_matrix;
           Assume eight direcitons are clock-wise ranging;

    output: weighted sum of acceleration by assigning 0.79 to major direction and 0.03 to splitted covariance acceleration

    '''

    new_acceleration = []

    for row in range(8):

        for col in range(8):

            if covariance_matrix[row, col] == 1:
                covariance_matrix[row, col] *= 0.79
            else:
                covariance_matrix[row, col] *= 0.03

    for i in range(8):

        new_acceleration.append(np.matmul(gradient.T, covariance_matrix[i, :]))

    return new_acceleration

def main():
    pass

if __name__ == '__main__':

    gradient = np.zeros(8)

    print(update_acceleration(gradient, covariance_matrix)[0])


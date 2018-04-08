'''
Design motion model for Particle Filter

'''

import os, sys
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

def main():
    pass

if __name__ == '__main__':
    main()


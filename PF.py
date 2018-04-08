## Particle Filter Implementation
'''
Data: 2018.04.02
Name: He Zhao
Inistitue: Yok Unidersity, Vision Lab
Advisor: Richard Wildes
Attention: -O flag to anneal __debug__ mode

'''
from __future__ import division
import numpy as np
from numpy.random import *
import os, sys
import scipy.io as io
from config import para, data_config
import matplotlib.pyplot as plt
import scipy.stats
import random
import math

class ParticleFilter(object):
	def __init__(self, **kwargs):
		# call back functions
		self.stateTransitFCN = kwargs['stateTransitFCN']
		self.observationFCN = kwargs['observationFCN']

		#process and measurement noise error
		self.errorPrior_mean = kwargs['errorPrior_mean']
		self.errorPrior_sigma = kwargs['errorPrior_sigma']
		self.error_normal_pdf = scipy.stats.norm(self.errorPrior_mean, self.errorPrior_sigma)

		# input data
		self.input_sequence = kwargs['input_sequence']
		self.highlight_period = kwargs['highlight_period']
		self.length = len(self.input_sequence)
		self.numOfParticle = kwargs['numOfParticle']

		# define particles prior and size
		self.particle_mean = kwargs['particle_mean']
		self.particle_sigma = kwargs['particle_sigma']
		self.particles = np.zeros((self.numOfParticle))
		self.particle_weights = np.zeros((self.numOfParticle))
		self.x_update = np.zeros((self.numOfParticle))
		self.particle_pred = np.zeros((self.length, self.numOfParticle))
		self.estimation = np.zeros(self.length)

		# init first batch of particles from random sampling of normal
		self.particles[:] = np.random.uniform(size=(1, self.numOfParticle))
		print("Init particle with uniform distribution in [0, 1]")

		# init first batch particles weights
		self.particle_weights[:] = 1 / self.numOfParticle

	def predict(self, time_idx):
		''' propagate particle foward with stateTransitFCN '''

		for i in range(self.numOfParticle):
			self.x_update[i] = self.stateTransitFCN(self.particles[i])

		self.particle_pred[time_idx, :] = self.x_update
		print("Range of predicted data value is {} to {}".format(np.min(self.x_update), np.max(self.x_update)))	

		# output prediction for this time step
		pred = np.mean(self.particle_pred[time_idx, :])
		
		return pred

	def update(self, time_idx, Observe=True):
		''' Update particles weights with Truth observation && Do resample to prevent from degendency'''

		# Using Observatio FCN
		if Observe:
			observation_input = self.observationFCN(self.input_sequence[time_idx]) + np.random.rand()
			obvervation_particle = self.observationFCN(self.x_update[:])
			diff = observation_input - obvervation_particle - 0.5
		
		else:
			diff = self.input_sequence[time_idx] - self.x_update

		# import pdb; pdb.set_trace()

		# calculate prob for particles
		self.particle_weights = (1.0 / math.sqrt(2*math.pi*self.errorPrior_sigma)) * np.exp(-(diff**2) / 2*self.errorPrior_sigma)
		# self.particle_weights = self.error_normal_pdf.pdf(diff)

		# norm prob weights
		self.particle_weights[:] = self.particle_weights[:] / (np.sum(self.particle_weights[:]) + 1e-10)

		# resample particles
		if 1.0 / np.sum(pow(self.particle_weights[:], 2)) < 0.5 * self.numOfParticle:
			print("Number of effective particle is {}".format(1.0 / np.sum(pow(self.particle_weights[:], 2))))
			print("Resampling ====================================")
			indices = self.resample(self.particle_weights)
			self.particles[:] = self.x_update[indices]
		else:
			self.particles = self.x_update

		# update prediction result and store into estimation
		pred = np.mean(self.particles)
		self.estimation[time_idx] = pred

		return pred

	def plotResult(self, ):
		pass

	def resample(self, weights):
		''' resample particle '''
		length = len(weights)
		indices = []

		# cumsum
		C = [0.] + [sum(weights[:i+1]) for i in range(length)]
		u0, j = np.random.rand(), 0

		# generate rising order random value to decide if discard weights
		b = [(u0+i)/length for i in range(length)]

		for u in b:
			while j < length and u > C[j]:
				j+=1
			indices.append(j-1)

		return indices

# callback functions
def stateTransitFCN(x, process_noise=1):
	a = abs(np.exp(x) + np.random.normal(0, 1))
	return a

def observationFCN(x):
	return x**2/0.5

if __name__ == '__main__':

	# load data
	file_path = os.path.join(data_config['root_dir'], data_config['data_name'])
	print(file_path)
	matFile = io.loadmat(file_path)
	data = matFile['result']

	# normalize data
	for i in range(12):
		data[i,:] /= np.max(data[i,:])

	# init para and PF
	para['stateTransitFCN'] = stateTransitFCN
	para['observationFCN'] = observationFCN
	para['input_sequence'] = data[2,:]
	pf = ParticleFilter(**para)

	# loop by time step, perform predict and update
	for j in range(len(data[7,:])):
		# import pdb; pdb.set_trace()
		pred = pf.predict(j)
		estimation = pf.update(j)	

	if __debug__:
		# plot particle and particle weights
		plt.figure(1)
		plt.plot(pf.particles[:])
		plt.figure(2)
		plt.plot(pf.particle_weights[:])
		plt.figure(3)
		plt.plot(pf.input_sequence)
		plt.show()

	plt.figure(1)
	plt.gca().set_color_cycle(['red', 'green', 'blue'])
	plt.plot(data[7,:])
	plt.plot(pf.estimation)
	plt.plot(np.mean(pf.particle_pred, axis=1))

	plt.legend(['True data', 'Estimation', 'Prediction'], loc='upper left')
	plt.show()















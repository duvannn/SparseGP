import numpy as np
from utils import *
from sys import argv
from math import pi, log
import scipy.optimize as opt
import pdb
# from <some kernel> import *

#########
# This is an implementation of the full Gaussian process 
#########

class fullGP():
	def __init__(self, N=100):
		self.trainX, self.trainY, self.testX, self.testY = get_all_data("kin40k")

		self.trainX = self.trainX[:N,:]
		self.trainY = self.trainY[:N,:]

	
	def train(self):
		N, D = self.trainX.shape
		# start guess
		x0 = 0.5*np.ones(D+2)
		# defining limits to the optimizer
		c_limit = (0.0000000000001, None)
		sigma2_limit = (0.0000000000001, None)
		b_limits = (0.0000000000001, None)
		bounds = [c_limit] + [sigma2_limit] + [b_limits]*D

		# now trying out the optimizer, using the parameters from sparse-GPSP by Mattias and Timo
		results = opt.minimize(fun = loglikelihood, 
								x0 = x0, 
								args = (self.trainX, self.trainY), 
								method = 'L-BFGS-B', 
								jac = gradients, 
								bounds = bounds)
		self.sigma2, self.c, self.b = params(results.x)

	#Return average of (prediction_mean_i - test_output_i)^2 for all test examples i
	#Prediction uses the mean of the predictive distribution p(y_star|x_star, y, X)
	def evaluateTestError(self):
		#Use predictive distribution for full GP
		K_N = kernelMatrix(self.trainX, self.trainX, self.c, self.b)
		K_N_inv = np.linalg.inv(K_N)

		predMatrix = np.dot(K_N_inv, self.trainY)

		#Get prediction of each test input
		sqTestErrorSum = 0.0
		for i, x_star in enumerate(self.testX):
			k = kernelMatrix(self.trainX, x_star, self.c, self.b)
			mean = np.dot(k.T, predMatrix)
			sqTestError = (mean - self.testY[i]) ** 2
			sqTestErrorSum += sqTestError

			if i % 1000 == 0:
				print str(i) + " predictions completed. Average error: " + str(sqTestErrorSum / (i + 1))
			
		return sqTestErrorSum / len(self.testX)

# Timo's kernel
#Kernel calculations
#KernelMatrix takes two matrices of data, can be both trainX or both pseudo or a mix.
def kernelMatrix(X,Y,c,b):
	scaledX = np.multiply(b, X) # elementwise mult: each dimension d is now (b^d)(x^d)
	scaledY = np.multiply(b, Y)
	scaledX2 = np.power(scaledX, 2) / b # each dimension d is now (b^d)^2(x^d)^2 / b^d
	scaledY2 = np.power(scaledY, 2) / b
	squared_distances = scaledX2.sum(1).reshape((scaledX.shape[0], 1)) + scaledY2.sum(1).reshape((1, scaledY.shape[0])) - 2 * scaledX.dot(Y.T)
	K = c*np.exp(-0.5*squared_distances)
	return K

#Log likelihood takes a 1D-array with the parameters
def loglikelihood(x, trainX, trainY):
	N, D = trainX.shape
	sigma, c, b = params(x)
	K_N = kernelMatrix(trainX, trainX, c, b)
	term = sigma*np.eye(N) + K_N
	inv_term = np.linalg.inv(term)

	sign, L_1 = np.linalg.slogdet(term)
	L_2 = np.dot(np.dot(np.transpose(trainY), inv_term), trainY).item(0)

	logLik = 0.5 * (L_1 + L_2 + N * log(2*pi))
	print logLik
	return logLik

def params(x0):
	# extracts the params and stores them into individual variables
	# params: 
	# x0 = an np.array with initial values of the parameters
	# 
	# extract sigma^2
	sigma = x0[0]
	# extract c
	c = x0[1]
	# extract b
	b = x0[2:]
	return sigma, c, b

def pack_params(sigma, c, b):
	# puts together parameters into an array
	temp_arr = np.array([sigma, c])
	return np.concatenate([temp_arr, b])

def kdot_wrt_b(X_d, K_N):
	# derivative of K_N wrt to b_d
	numerator = np.power(X_d - X_d.reshape(1, X_d.shape[0]), 2)
	return -0.5 * np.multiply(numerator, K_N)

def grad_b(trainX, trainY, K_N, inv_term):
	N, D = trainX.shape
	# calculates the gradient wrt b
	b_arr = []
	for i in range(D):
		# compute derivative of K_N wrt to b_d
		X_d = trainX[:,i]
		K_dot = kdot_wrt_b(X_d, K_N)

		L_1 = np.trace( np.dot(inv_term, K_dot))
		L_2 = - np.dot( np.dot(trainY.T, np.dot( np.dot(inv_term, K_dot), inv_term)), trainY)
		b_arr.append( 0.5*(L_1 + L_2).item(0))

	return np.array(b_arr)

def grad_c(trainY, c, K_N, inv_term):
	# calculates the gradient wrt to c
	K_dot = (1/c)*K_N
	L_1 = np.trace( np.dot(inv_term, K_dot))
	L_2 = - np.dot( np.dot(trainY.T, np.dot( np.dot(inv_term, K_dot), inv_term)), trainY)
	return 0.5*(L_1 + L_2).item(0)

def grad_sigma(trainY, sigma, term, inv_term):
	# calculates the gradient wrt to sigma
	L_1 = np.trace(inv_term)
	inv_term = np.linalg.inv(np.dot(term, term))
	L_2 = -np.dot( np.dot( trainY.T, inv_term), trainY)
	return 0.5*(L_1 + L_2).item(0)

def gradients(x, trainX, trainY):
	# computes the different gradients and stores them in an array
	# params:
	# x = parameters [sigma^2, c, b_1, ..., b_D]
	N, D = trainX.shape
	sigma, c, b = params(x)
	K_N = kernelMatrix(trainX, trainX, c, b)
	term = sigma*np.eye(N) + K_N
	inv_term = np.linalg.inv(term)

	sigma_g = grad_sigma(trainY, sigma, term, inv_term)
	c_g = grad_c(trainY, c, K_N, inv_term)
	b_g = grad_b(trainX, trainY, K_N, inv_term)

	#print pack_params(sigma_g, c_g, b_g)
	return pack_params(sigma_g, c_g, b_g)
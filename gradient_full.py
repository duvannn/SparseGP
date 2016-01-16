import numpy as np
from utils import *
from sys import argv
from math import pi, log
# from <some kernel> import *

#########

# This is an implementation of the full Gaussian process 

#########


trainX, trainY, testX, testY = get_all_data("kin40k")

print trainX

print trainX.shape
print trainY.shape
print testX.shape
print testY.shape

Y = trainY

# define D
D = trainX.shape[1]
# define N
N = trainX.shape[0]

# start guess
x0 = 0.5*np.ones(D+2)
params0 = x0
## IMPROVEMENTS:

# make K_N, term, inv_term acessible to all the functions




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
def loglikelihood(x):
	sigma, c, b = params(x)
	K_N = kernelMatrix(trainX, trainX, c, b)
	term = sigma*np.eye(N) + K_N
	inv_term = np.linalg.inv(term)
	L_1 = log(np.linalg.det(term))
	L_2 = np.dot(np.dot(np.transpose(Y), inv_term), Y).item(0)
	return 0.5 * (L_1 + L_2 + log(2*pi))

# get L_1 = -inf 

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
	b = x0[2:D+2]
	return sigma, c, b

def pack_params(sigma, c, b):
	temp_arr = np.array([sigma, c])
	return np.concatenate([temp_arr, b])

def kdot_wrt_b(X_d,X_dprime, K_N):
	K_bdot = np.power(X_d - X_dprime.reshape(1,N), 2)
	return -0.5*(np.dot(K_bdot, K_N))

def grad_b(Xtrain, K_N, term, inv_term):
	# currently this does not work
	b_arr = []
	for i in range(D):
		# this is incorrect!
		X_d = Xtrain[:,i]
		K_dot = kdot_wrt_b(X_d, X_d, K_N)
		L_1 = np.trace(inv_term)*K_dot
		L_2 = -np. dot ( np. dot (Y.T, np.dot(L_1, term)), Y)
		b_arr.append( 0.5*(L_1 + L_2).item(0) )
	return np.array(b_arr)

def grad_c(c,K_N, term, inv_term):
	K_dot = (1/c)*K_N
	L_1 = np.trace(inv_term)*K_dot
	L_2 = - np. dot ( np.dot(Y.T, np.dot( np.dot(term, K_dot), term)), Y)
	return 0.5*(L_1 + L_2).item(0)

def grad_sigma(sigma,K_N, term):
	L_1 = np.trace(term)
	L_2 = np. dot( np.dot( Y.T, np.power(term, 2)), Y)
	return 0.5*(L_1 + L_2).item(0)


def gradients(x):
	sigma, c, b = params(x)
	K_N = kernelMatrix(trainX, trainX, c, b)
	term = sigma*np.eye(N) + K_N
	inv_term = np.linalg.inv(term)
	sigma_g = grad_sigma(sigma, K_N, term)
	c_g = grad_c(c,K_N, term, inv_term)
	b_g = grad_b(Xtrain, K_N, term, inv_term)
	return pack_params(sigma_g, c_g, b_g)


# optimizer NEEDS TESTING, IMPROVEMENT etc
results = opt.minimize(fun = loglikelihood, 
								x0 = params0,
								args = (trainX, trainY), 
								method = 'L-BFGS-B',
								jac = gradient)

sigma_opt, c_opt, b_opt = pack_params(results)
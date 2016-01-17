import numpy as np
from utils import *
from sys import argv
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


#Log likelihood takes a 1D-array with the parameters
def loglikelihood(x):
	sigma, c, b = params(x)

	N = X.shape[0] # X is NxD
	K_N = kernel(trainX, trainX, c, b)
	term = np.diag([sigma] * N) + K_N
	inv_term = np.linalg.inv(term)
	L_1 = math.log(np.linalg.det(term))
	L_2 = np.dot(np.dot(np.transpose(Y), inv_term), Y)
	return 0.5 * (L_1 + L_2 + N * math.log(2*math.pi))

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

def gradient(x0):
	sigma, c, b = params(x0)

	# gradient wrt b

	# gradient wrt c

	# gradient wrt sigma^2

	#return gradient_vec
	return 0

# optimizer

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

# define D
D = trainX.shape[1]
# define N
N = trainX.shape[0]


def loglikelihood():
	return 0

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

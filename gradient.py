import numpy as np
from utils import *

trainX, trainY, testX, testY = get_all_data("kin40k")

print trainX

print trainX.shape
print trainY.shape
print testX.shape
print testY.shape


def kernel():
	return 0


def loglikelihood(M=None):
	return 0


def gamma_prime():
	return 0

def gradsum():
	fi1_p = 
	return 0

def params(x0, SPGP=False):
	# extracts the params and stores them into individual variables
	# params: 
	# x0 = array with initial values of the parameters
	# 
	# extract sigma^2
	sigma = x0[0]
	# extract c
	c = x0[1]
	# extract b
	b = x0[2:D+2]
	if SPGP==True:
		# extract pseudo-inputs
		X = x0[D+3:]
		return sigma, c, b, X
	return sigma, c, b

def gradient(SPGP=False, M=None):
	#if SPGP == True:
		# gradient_vec.append()
		# compute gradient wrt pseudo-inputs

	# gradient wrt b


	# gradient wrt c

	# gradient wrt sigma^2

	#return gradient_vec
	return 0

# optimizer

import numpy as np
import math
from utils import *

trainX, trainY, testX, testY = get_all_data("kin40k")

print trainX

print trainX.shape
print trainY.shape
print testX.shape
print testY.shape

Y = trainY

def kernel():
	return 0


#Log likelihood takes a 1D-array with the parameters
def loglikelihood(x): 
	if M:
		#SPGP case
		sigma, c, b, X = params(x)
		#TBD
		return

	#full GP case
	sigma, c, b = params(x)

	N = X.shape[0] # X is NxD
	K_N = kernel(trainX, trainX, c, b)
	term = np.diag([sigma] * N) + K_N
	inv_term = np.linalg.inv(term)
	L_1 = math.log(np.linalg.det(term))
	L_2 = np.dot(np.dot(np.transpose(Y), inv_term), Y)
	return 0.5 * (L_1 + L_2 + N * math.log(2*math.pi))


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

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


def loglikelihood():
	return 0


def gradient(SPGP=False):
	#if SPGP == True:
		# gradient_vec.append()
		# compute gradient wrt pseudo-inputs

	# gradient wrt b

	# gradient wrt c

	# gradient wrt sigma^2

	#return gradient_vec
	return 0

# optimizer

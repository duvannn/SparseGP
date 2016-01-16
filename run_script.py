from utils import get_all_data, read_dataset, mean_square_error,generateSinData
from gp import GaussianProcess, SparseGaussianProcess
from kernels import ARDKernel
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T

def main(iters = 1000, points = 1000,pseudos = 50):
	print "running"
	trainX, trainY, testX, testY = get_all_data("kin40k")
	sgp = SparseGaussianProcess(np.array(trainX[0:points,:],dtype=np.float64), np.array(trainY[0:points,:],dtype=np.float64),pseudos)
	rms = sgp.RMSprop()
	cost = []
	print "training begun"
	for i in range(iters):
		c = rms()
		cost.append(c)
	pred= sgp.getPosteriorPredictive()
	mu,cov = pred(testX[:1000])
	print mean_square_error(mu,testY[:1000])
if __name__ == "__main__": main()
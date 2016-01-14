import numpy as np
import math
import random
from utils import *
from sys import argv

#params structure [c, sigma^2, b_1,...,b_D, x_1^1,...,x_D^1, ..., x_1^M,...,x_D^M]
class gp():

	#limit is used temporarily to limit n.o training points
	def __init__(self, M = 10, limit = 100):
		trX, trY, tX, tY = get_all_data("kin40k")
		self.trainX = trX[:limit] if limit else trX
		self.trainY = trY[:limit] if limit else trY
		self.testX = tX
		self.testY = tY

		#Get dataset dimensions
		self.N, self.D = self.trainX.shape
		self.M = M

	#TODO: Initiate scipy optimizer
	def run(self):
		params0 = self.get_initial_params_guess()	
		return neg_loglikelihood(params0, self.trainX, self.trainY, self.M)
	
	#Initial guess of pseudo inputs is random subset of training inputs
	def get_initial_params_guess(self):
		c = 0.5
		sigma2 = 0.5 
		b = np.random.random((1,self.D))
		Xbar = self.trainX[random.sample(range(0, self.trainX.shape[0]), self.M),:]
		return pack_variables(c, sigma2, b, Xbar)

#Kernel takes two matrices of data, can be both trainX or both pseudo or a mix.
def kernelMatrix(X,Y,c,b): 
    scaledX = np.multiply(np.sqrt(b),X)
    scaledY = np.multiply(np.sqrt(b),Y)
    squared_distances = np.power(scaledX, 2).sum(1).reshape((scaledX.shape[0], 1)) + (np.power(scaledY, 2)).sum(1).reshape((1, scaledY.shape[0])) - 2 * scaledX.dot(scaledY.T)
    K = c*np.exp(-0.5*squared_distances)
    return K

def kernel(x_1, x_2, c, b):
	return c * math.exp( -0.5 * np.sum(b * (np.array(x_1 - x_2) ** 2)))

def unpack_variables(params, D, M):
	c = params[0]
	sigma2 = params[1]
	b = params[2:2+D]
	Xbar = params[2+D:].reshape(M,D)
	return c, sigma2, b, Xbar

def pack_variables(c, sigma2, b, Xbar):
	b_list = b.tolist()[0]
	Xbar_list = Xbar.reshape(1, Xbar.shape[0] * Xbar.shape[1]).tolist()[0]
	return np.array([c, sigma2] + b_list + Xbar_list)

def neg_loglikelihood(params, X, y, M):
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	
	#TODO: update with less expensive calculations
	K_NM = kernelMatrix(X,Xbar,c,b)
	K_M = kernelMatrix(Xbar,Xbar,c,b)
	K_M_inv = np.linalg.inv(K_M)
	K_MN = K_NM.T

	s2G = sigma2Gamma(sigma2, X, Xbar, K_M_inv, c, b)

	innerMatrix = s2G + np.dot(np.dot(K_NM, K_M_inv), K_MN)
	phi_1 = math.log(np.linalg.det(innerMatrix))
	phi_2 = np.dot(np.dot(y.T, np.linalg.inv(innerMatrix)), y)

	return (0.5 * (phi_1 + phi_2 + N * math.log(2*math.pi)))[0,0]

def sigma2Gamma(sigma2, X, Xbar, K_M_inv, c, b):
	N = X.shape[0]
	sigma2_I = np.zeros((N,N))
	np.fill_diagonal(sigma2_I, sigma2)

	Lambda = np.zeros((N,N))
	for n in range(0,N):
		x_n = X[n,:]
		K_nn = kernel(x_n, x_n, c, b)
		k_x_n = kernelMatrix(Xbar, x_n, c, b)
		f = np.dot(k_x_n.T, K_M_inv)
		Lambda[n][n] =  K_nn - np.dot(f, k_x_n)[0,0]

	return sigma2_I + Lambda

def gradient(params, X, y, M):
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	gradList = [None for i in range(0, params)]
	gradList[0] = gradient_wrt_c()
	gradList[1] = gradient_wrt_sigma2()
	gradList[2:2+D] = gradient_wrt_b()
	gradList[2+D:] = gradient_wrt_Xbar()
	return gradList

def gradient_wrt_c():
	pass

def gradient_wrt_sigma2():
	pass

def gradient_wrt_b():
	pass

def gradient_wrt_Xbar():
	pass

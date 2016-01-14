import numpy as np
import math
import random
from utils import *
from sys import argv

#params structure [c, sigma^2, b_1,...,b_D, xbar_1^1,...,xbar_1^D, ..., xbar_M^1,...,xbar_M^D]
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

#Kernel calculations
#KernelMatrix takes two matrices of data, can be both trainX or both pseudo or a mix.
def kernelMatrix(X,Y,c,b): 
	scaledX = np.multiply(np.sqrt(b),X)
	scaledY = np.multiply(np.sqrt(b),Y)
	squared_distances = np.power(scaledX, 2).sum(1).reshape((scaledX.shape[0], 1)) + (np.power(scaledY, 2)).sum(1).reshape((1, scaledY.shape[0])) - 2 * scaledX.dot(scaledY.T)
	K = c*np.exp(-0.5*squared_distances)
	return K

def kernel(x_1, x_2, c, b):
	return c * math.exp( -0.5 * np.sum(b * (np.array(x_1 - x_2) ** 2)))

#Vectorizing and unvectorizing variables for use in scipy optimizer
def pack_variables(c, sigma2, b, Xbar):
	b_list = b.tolist()[0]
	Xbar_list = Xbar.reshape(1, Xbar.shape[0] * Xbar.shape[1]).tolist()[0]
	return np.array([c, sigma2] + b_list + Xbar_list)

def unpack_variables(params, D, M):
	c = params[0]
	sigma2 = params[1]
	b = params[2:2+D]
	Xbar = params[2+D:].reshape(M,D)
	return c, sigma2, b, Xbar

#Neg log likelihood calculations
def neg_loglikelihood(params, X, y, M):
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	
	#TODO: update with less expensive calculations
	K_NM = get_K_NM(X, Xbar, c, b)
	K_M = get_K_M(Xbar, c, b)
	K_M_inv = np.linalg.inv(K_M)
	s2G = sigma2 * get_Gamma(sigma2, X, Xbar, K_M_inv, c, b)

	innerMatrix = s2G + np.dot(np.dot(K_NM, K_M_inv), K_NM.T)
	phi_1 = math.log(np.linalg.det(innerMatrix))
	phi_2 = np.dot(np.dot(y.T, np.linalg.inv(innerMatrix)), y)

	return (0.5 * (phi_1 + phi_2 + N * math.log(2*math.pi)))[0,0]

#Gradient calculations
def gradient(params, X, y, M):
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	
	g = generalGradientVars(X, Xbar, y, sigma2, c, b, M)

	gradList = [None for i in range(0, params)]
	gradList[0] = gradient_wrt_c(X, Xbar, y, sigma2, c, b, M, g)
	gradList[1] = gradient_wrt_sigma2()
	gradList[2:2+D] = gradient_wrt_b()
	gradList[2+D:] = gradient_wrt_Xbar()
	return gradList

def gradient_wrt_c(X, Xbar, y, sigma2, c, b, M, g):
	v = variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, 0)

	pd1 = phiDot_1(g, 'c')
	pd2 = phiDot_2()
	return 0.5 * (pd1 + pd2)

def gradient_wrt_sigma2(params, X, y, M): #phi_1 and phi_2 will be calculated here explicitely since it is of different form than for the other gradients 
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	K_NM = get_K_NM(X, Xbar, c, b)
	K_M = get_K_M(Xbar, c, b)
	K_M_inv = np.linalg.inv(K_M)
	K_MN = K_NM.T
	Gamma = get_Gamma(sigma2, X, Xbar, K_M_inv, c, b)
	Gamma_inv = np.linalg.inv(Gamma)
	A = get_A(sigma2, K_M, K_NM, Gamma)

	sigma2_inv = math.pow(sigma2, -1)
	Z = np.dot(K_NM,np.dot(np.linalg.inv(A),K_MN)) #auxiliery 
	U = y.T.dot(Gamma_inv.dot(Z.dot(np.power(Gamma_inv,2).dot(y)))) #auxiliery 
	phi_1 = sigma2_inv*np.matrix.trace(Gamma_inv) - sigma2_inv * np.matrix.trace(np.dot(Gamma_inv,np.dot(Z,Gamma_inv)))
	phi_2 = -sigma2_inv**2*(np.linalg.norm(np.dot(Gamma_inv,y))**2 + np.linalg.norm(Gamma_inv.dot(np.dot(Z,np.dot(Gamma_inv,y))))**2-U-U.T)
	return 0.5 * phi_1 + 0.5 * phi_2

def gradient_wrt_b():
	pass

def gradient_wrt_Xbar():
	pass

#Derivatives of phi_1 and phi_2 wrt the specified varName variable
def phiDot_1(gMs):
	



	Gamma_term


	K_term
	return np.trace(A_term) + np.trace(Gamma_term) - np.trace(K_term)


def phiDot_2():
	pass

#Calculations of matrices and vectors used by gradients in a dict
#General: used by all gradients
def generalGradientVars(X, Xbar, y, sigma2, c, b, M):
	g = {}

	g['K_M'] = get_K_M(Xbar, c, b)
	g['K_NM'] = get_K_NM(X, Xbar, c, b)
	g['Gamma'] = get_Gamma(sigma2, X, Xbar, np.linalg.inv(g['K_M']), c, b)
	g['A'] = get_A(sigma2, g['K_M'], g['K_NM'], g['Gamma'])
	
	g['A_half'] = np.linalg.cholesky(g['A']) 
	g['Gamma_half'] = np.linalg.cholesky(g['Gamma'])
	g['K_M_half'] = np.linalg.cholesky(g['K_M'])

	return g

#Variable specific: different when taking gradient wrt different variables
def variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, varIndex):
	v = {}

	v['K_NM_dot'] = get_K_NM_dot(X, Xbar, c, b, g, varIndex)
	v['K_NM_bar_dot'] = 
	v['K_M_dot'] = 
	v['K_NM_bar'] = 
	v['Gamma_bar_dot'] =
	v['A_dot'] = 

	return v

#Matrix and vector calculations used by neg log likelihood and gradient calculations
def get_K_M(Xbar, c, b):
	return kernelMatrix(Xbar,Xbar,c,b)

def get_K_NM(X, Xbar, c, b):
	return kernelMatrix(X,Xbar,c,b)

def get_Gamma(sigma2, X, Xbar, K_M_inv, c, b):
	N = X.shape[0]
	I = np.identity(N)
	
	Lambda = np.zeros((N,N))
	for n in range(0, N):
		x_n = X[n,:]
		K_nn = kernel(x_n, x_n, c, b)
		k_x_n = kernelMatrix(Xbar, x_n, c, b)
		f = np.dot(k_x_n.T, K_M_inv)
		Lambda[n][n] =  K_nn - np.dot(f, k_x_n)[0,0]

	return I + Lambda / sigma2

#TODO: better inversion of Gamma?
def get_A(sigma2, K_M, K_NM, Gamma):
	return sigma2 * K_M + np.dot(np.dot(K_NM.T, np.linalg.inv(Gamma)), K_NM)

def get_K_NM_dot(X, Xbar, c, b, varIndex):
	M, D = Xbar.shape

	if varIndex == 0: #c
		return g['K_NM'] / c

	#elif varIndex == 1:  #gradient wrt sigma^2 calculated differently

	varIndex -= 2 #b indices start at 2
	elif varIndex < D: #b_1 to b_D
		
		#Extract relevant dimension in X and Xbar.
		X_dim = X[:,varIndex]
		Xbar_dim = Xbar[:,varIndex]

		#Calculate gradient numerator
		numerator = (X_dim - Xbar_dim.reshape(1, Xbar_dim.shape[0])) ** 2
		return -0.5 * numerator * g['K_NM']

		#################
 		K_N = get_K_M(X, c, b)
		K_NM_dot_list = [None for i in range(0, D)]
		for d in range(0,D):                        
			Distances = np.power(X[:,d], 2).reshape((X.shape[0], 1)) + np.power(X[:,d], 2).reshape((1, X.shape[0])) - 2 * X[:,d].reshape((X.shape[0], 1)) * X[:,d].reshape((1, X.shape[0]))
			K_NM_dot_list[d] = -0.5*np.multiply(Distances,K_N)
                 
 
 	#else xbar_1^1 to xbar_D^M
	return K_NM_dot_list
	#################

	#else xbar_1^1 to xbar_M^D
	varIndex -= D #xbar indices start at 2+D
	m = varIndex / D #Get the pseudo input index, i.e. m in xbar_m (0 to M-1)
	d = varIndex % D #Get the relevant dimension, i.e. d in xbar_m^d (0 to D-1)
	X_dim = X[:,d]
	Xbar_dim = Xbar[:,d]

	#Calculate gradient differrence term
	diffTerm = X_dim - Xbar_dim.reshape(1, Xbar_dim.shape[0])

	#columns different than ones relating to x_m should be 0
	rowVector = np.zeros((1,M))
	rowVector[0][m] = 1
	diffTerm = diffTerm * rowVector #Will set all other cols than m to 0

	return b[d] * diffTerm * g['K_NM']



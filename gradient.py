import numpy as np
import scipy as sp
import scipy.optimize as opt
import math
import random
from utils import *
from sys import argv

#params structure [c, sigma^2, b_1,...,b_D, xbar_1^1,...,xbar_1^D, ..., xbar_M^1,...,xbar_M^D]
class spgp():

	c_limit = (0, None)
	sigma2_limit = (0.0000000000001, None)
	b_limits = (0.0000000000001, None)
	xbar_limits = (None, None)

	#limit is used temporarily to limit n.o training points
	def __init__(self, limit = 30):
		trX, trY, tX, tY = get_all_data("kin40k")
		self.trainX = trX[:limit] if limit else trX
		self.trainY = trY[:limit] if limit else trY
		self.testX = tX
		self.testY = tY
		self.N, self.D = self.trainX.shape #Get dataset dimensions

	#Train SPGP
	def train(self, M = 2):
		self.M = M
		params0 = self.get_initial_params_guess()
		bounds = [self.c_limit] + [self.sigma2_limit] + [self.b_limits] * self.D + [self.xbar_limits] * self.M * self.D
		results = opt.minimize(fun = neg_loglikelihood, 
								x0 = params0,
								args = (self.trainX, self.trainY, self.M), 
								method = 'L-BFGS-B',
								jac = gradient, 
								bounds = bounds)
		self.c, self.sigma2, self.b, self.Xbar = unpack_variables(results.x, self.D, self.M)

	#Return average of (prediction_mean_i - test_output_i)^2 for all test examples i
	#Prediction uses the mean of the predictive distribution p(y_star|x_star, y, X, Xbar)
	def evaluateTestError(self):
		g = generalGradientVars(self.trainX, self.Xbar, self.trainY, self.sigma2, self.c, self.b, self.M, True)
		LambdaAndSigma2_inv = np.linalg.inv(g['Gamma'] * self.sigma2)
		Q_M = g['K_M'] + np.dot(g['K_NM'].T, np.dot(LambdaAndSigma2_inv, g['K_NM']))
		Q_M_inv = np.linalg.inv(Q_M)
		predMatrix = np.dot(Q_M_inv, np.dot(g['K_NM'].T, np.dot(LambdaAndSigma2_inv, self.trainY)))

		#Get prediction of each test input
		sqTestErrorSum = 0.0
		for i, x_star in enumerate(self.testX):
			k_star = get_k_star(x_star, self.Xbar, self.c, self.b)
			mu_star = np.dot(k_star.T, predMatrix)
			sqTestError = (mu_star - self.testY[i]) ** 2
			sqTestErrorSum += sqTestError

			if i % 1000 == 0:
				print str(i) + " predictions completed. Average error: " + str(sqTestErrorSum / (i + 1))
			
		return sqTestErrorSum / len(self.testX)

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
	scaledX = np.multiply(b, X) # elementwise mult: each dimension d is now (b^d)(x^d)
	scaledY = np.multiply(b, Y)
	scaledX2 = np.power(scaledX, 2) / b # each dimension d is now (b^d)^2(x^d)^2 / b^d
	scaledY2 = np.power(scaledY, 2) / b

	squared_distances = scaledX2.sum(1).reshape((scaledX.shape[0], 1)) + scaledY2.sum(1).reshape((1, scaledY.shape[0])) - 2 * scaledX.dot(Y.T)
	K = c*np.exp(-0.5*squared_distances)
	return K

#(actually only used when sending in identical inputs, so the output is always c)
def kernel(x_1, x_2, c, b):
	return c * math.exp( -0.5 * np.sum(np.multiply(b, np.power(x_1 - x_2, 2))))

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
	g = generalGradientVars(X, Xbar, y, sigma2, c, b, M, True)

	#For Gamma, find log of the determinant to prevent float overflow issues
	detSign, logDet = np.linalg.slogdet(g['Gamma'])
	if detSign < 0:
		print "Gamma's determinant was < 0!"

	phi_1 = math.log(np.linalg.det(g['A'])) + logDet - math.log(np.linalg.det(g['K_M'])) 
	phi_1 += (N-M) * math.log(sigma2)

	innerMatrix = g['Gamma_inv'] - np.dot(g['Gamma_inv'], np.dot(g['K_NM'], np.dot(g['A_inv'], np.dot(g['K_NM'].T, g['Gamma_inv']))))
	phi_2 = (sigma2 ** (-1)) * np.dot(y.T, np.dot(innerMatrix, y))

	loglikelihood = (0.5 * (phi_1 + phi_2 + N * math.log(2*math.pi)))[0,0]
	print "loglikelihood: " + str(loglikelihood)
	return loglikelihood

#Gradient calculations
def gradient(params, X, y, M):
	N, D = X.shape
	c, sigma2, b, Xbar = unpack_variables(params, D, M)
	g = generalGradientVars(X, Xbar, y, sigma2, c, b, M)

	gradList = np.array([0.0 for i in range(0, params.shape[0])])
	gradList[0] = gradient_wrt_c(X, Xbar, y, sigma2, c, b, M, g)
	gradList[1] = gradient_wrt_sigma2(y, sigma2, g)
	gradList[2:2+D] = gradient_wrt_b(X, Xbar, y, sigma2, c, b, M, g)
	gradList[2+D:] = gradient_wrt_Xbar(X, Xbar, y, sigma2, c, b, M, g)
	return gradList

def gradient_wrt_c(X, Xbar, y, sigma2, c, b, M, g):
	return gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, 0)

def gradient_wrt_b(X, Xbar, y, sigma2, c, b, M, g):
	D = X.shape[1]
	gradVec = [0.0 for i in range(0, D)]
	for i in range(0,D):
		gradVec[i] = gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, 2+i)
	return gradVec

def gradient_wrt_Xbar(X, Xbar, y, sigma2, c, b, M, g):
	D = X.shape[1]
	gradVec = [0.0 for i in range(0, M*D)]
	for i in range(0, M*D):
		gradVec[i] = gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, 2+D+i)
	return gradVec

#phi_1 and phi_2 will be calculated here explicitely since it is of different form than for the other gradients
def gradient_wrt_sigma2(y, sigma2, g):
	sigma2_inv = math.pow(sigma2, -1)
	Z = np.dot(g['K_NM'], np.dot(g['A_inv'],g['K_NM'].T)) #auxiliery 
	U = y.T.dot(g['Gamma_inv'].dot(Z.dot(np.power(g['Gamma_inv'],2).dot(y)))) #auxiliery 
	
	phi_1 = sigma2_inv * np.trace(g['Gamma_inv']) - sigma2_inv * np.trace(np.dot(g['Gamma_inv'],np.dot(Z,g['Gamma_inv'])))
	#phi_2 = -(sigma2_inv**2)*(np.linalg.norm(np.dot(g['Gamma_inv'],y))**2 + np.linalg.norm(g['Gamma_inv'].dot(np.dot(Z,np.dot(g['Gamma_inv'],y))))**2-U-U.T)
	phi_2 = -(sigma2_inv**2) * np.linalg.norm(np.dot(g['Gamma_inv'] - np.dot(g['Gamma_inv'], np.dot(g['K_NM'], np.dot(g['A_inv'], np.dot(g['K_NM'].T, g['Gamma_inv'])))), y))**2
	return 0.5 * phi_1 + 0.5 * phi_2

def gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, index):
	v = variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, index)
	pd1 = phiDot_1(g, v)
	pd2 = phiDot_2(sigma2, g, v)
	return 0.5 * (pd1 + pd2)

#Derivatives of phi_1 and phi_2, using general info in g and variable specific info in v
#assuming that in derivations A^(T/2) means A^(1/2).T
def phiDot_1(g, v):
	A_term = g['A_half_inv'].dot(v['A_dot'].dot(g['A_half_inv'].T)) 
	Gamma_term = v['Gamma_bar_dot']
	K_term = g['K_M_half_inv'].dot(v['K_M_dot'].dot(g['K_M_half_inv'].T))
	return np.trace(A_term) + np.trace(Gamma_term) - np.trace(K_term)

def phiDot_2(sigma2, g, v):
	Term1 = - g['y_Gamma'].T.dot(v['Gamma_bar_dot'].dot(g['y_Gamma']))
	Term2 = 2 * g['y_Gamma'].T.dot(v['Gamma_bar_dot'].dot(g['K_NM_bar'].dot(g['A_inv'].dot(g['K_NM_bar'].T.dot(g['y_Gamma'])))))
	Term3 = -2 * g['y_Gamma'].T.dot(g['K_NM_bar'].dot(g['A_inv'].dot(v['K_NM_bar_dot'].T.dot(g['y_Gamma']))))
	Term4 = g['y_Gamma'].T.dot(g['K_NM_bar'].dot(g['A_inv'].dot(v['A_dot'].dot(g['A_inv'].dot(g['K_NM_bar'].T.dot(g['y_Gamma']))))))
	return sigma2**(-1) * (Term1 + Term2 + Term3 + Term4)

#Calculations of matrices and vectors used by gradients in a dict
#General: used by all gradients
#also used in calculation of loglikelihood, but we don't need to include all calculations.
def generalGradientVars(X, Xbar, y, sigma2, c, b, M, limitCalcs = False):
	g = {}

	g['K_M'] = get_K_M(Xbar, c, b)
	g['K_M_inv'] = np.linalg.inv(g['K_M'])
	g['K_NM'] = get_K_NM(X, Xbar, c, b)

	g['Gamma'] = get_Gamma(sigma2, X, Xbar, g['K_M_inv'], g['K_NM'], c, b)
	g['Gamma_inv'] = np.linalg.inv(g['Gamma'])
	g['A'] = get_A(sigma2, g['K_M'], g['K_NM'], g['Gamma_inv']) 
	g['A_inv'] = np.linalg.inv(g['A'])

	if limitCalcs: return g
	
	g['A_half'] = np.linalg.cholesky(g['A'])
	g['A_half_inv'] = np.linalg.inv(g['A_half'])
	g['Gamma_half'] = np.linalg.cholesky(g['Gamma'])
	g['Gamma_half_inv'] = np.linalg.inv(g['Gamma_half'])
	g['K_M_half'] = np.linalg.cholesky(g['K_M'])		
	g['K_M_half_inv'] = np.linalg.inv(g['K_M_half'])

	g['K_NM_bar'] = np.dot(g['Gamma_half_inv'], g['K_NM'])
	g['y_Gamma'] = g['Gamma_half_inv'].dot(y)

	return g

#Variable specific: different when taking gradient wrt different variables
def variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, varIndex):
	v = {}

	v['K_NM_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'NM')
	v['K_NM_bar_dot'] = np.dot(g['Gamma_half_inv'], v['K_NM_dot'])
	v['K_M_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'M')
	
	v['Gamma_dot'] = get_Gamma_dot(X, c, sigma2, b, g, v, varIndex)
	v['Gamma_bar_dot'] = np.dot(np.dot(g['Gamma_half_inv'], v['Gamma_dot']), g['Gamma_half_inv'])
	v['A_dot'] = get_A_dot(sigma2, g, v)

	return v

#Matrix and vector calculations used by neg log likelihood and gradient calculations
def get_K_M(Xbar, c, b):
	return kernelMatrix(Xbar,Xbar,c,b)

def get_K_NM(X, Xbar, c, b):
	return kernelMatrix(X,Xbar,c,b)

def get_Gamma(sigma2, X, Xbar, K_M_inv, K_NM, c, b):
	N = X.shape[0]
	I = np.identity(N)

	Lambda = np.zeros((N,N))
	#for n in range(0, N):
	#	x_n = X[n,:]
	#	K_nn = kernel(x_n, x_n, c, b)
	#	k_x_n = kernelMatrix(Xbar, x_n, c, b)			
	#	Lambda[n][n] =  K_nn - np.dot(np.dot(k_x_n.T, K_M_inv), k_x_n)[0,0]
	L = np.zeros((N,N))
	np.fill_diagonal(L,np.diagonal(np.dot(np.dot(K_NM,K_M_inv),K_NM.T)))
        Lambda = c * np.identity(N) - L
	return I + Lambda / sigma2

#TODO: better inversion of Gamma?
def get_A(sigma2, K_M, K_NM, Gamma_inv):
	return sigma2 * K_M + np.dot(np.dot(K_NM.T, Gamma_inv), K_NM)

#Returns a gradient matrix K_NM_dot or K_M_dot
#varIndex: the index of the relevant variable in the params vector
#subscript should be either of:
#'NM': to return K_NM_dot (row inputs are training points)
#'M': to return K_M_dot (row inputs are pseudo points)
def get_K_dot(X, Xbar, c, b, g, varIndex, subscript):
	M, D = Xbar.shape

	#Differentiate between K_NM and K_M
	dictKey = 'K_' + subscript
	rowInputs = X if subscript == 'NM' else Xbar

	if varIndex == 0: #c
		return g[dictKey] / c

	#elif varIndex == 1:  #gradient wrt sigma^2 calculated differently

	varIndex -= 2 #b indices start at 2
	if varIndex < D: #b_1 to b_D
		
		#Extract relevant dimension in X and Xbar.
		rowInput_dim = rowInputs[:,varIndex]
		Xbar_dim = Xbar[:,varIndex]

		#Calculate gradient numerator
		numerator = np.power(rowInput_dim - Xbar_dim.reshape(1, Xbar_dim.shape[0]), 2)
		return -0.5 * np.multiply(numerator, g[dictKey])

	#else xbar_1^1 to xbar_M^D
	varIndex -= D #xbar indices start at 2+D
	m = varIndex / D #Get the pseudo input index, i.e. m in xbar_m (0 to M-1)
	d = varIndex % D #Get the relevant dimension, i.e. d in xbar_m^d (0 to D-1)
	rowInput_dim = rowInputs[:,d]
	Xbar_dim = Xbar[:,d]

	#Calculate gradient differrence term
	diffMatrix = rowInput_dim - Xbar_dim.reshape(1, Xbar_dim.shape[0])

	#In K_NM_dot columns different than ones relating to x_m should be 0
	rowVector = np.zeros((1, M))
	rowVector[0][m] = 1
	diff = np.multiply(diffMatrix, rowVector) #Will set all other cols than m to 0

	#In K_M_dot, the rows also represent pseudo inputs, so the m:th row should not be 0
	if subscript == 'M':
		colVector = np.zeros((M, 1))
		colVector[m][0] = 1
		diff2 = diffMatrix * rowVector
		diff = diff + diff2 #elementwise addition (will keep the m,m th element 0) 

	return b[d] * np.multiply(diff, g[dictKey]) #elementwise multiplication

#varIndex: the index of the relevant variable in the params vector
def get_Gamma_dot(X, c, sigma2, b, g, v, varIndex):
	N, D = X.shape

	Gamma_dot = np.zeros((N,N))
	#for n in range(0,N):
	#	K_nn_dot = get_K_nn_dot(X, c, b, n, varIndex)
	#	k_n_dot = get_k_n_dot(v, n)
	#	k_n = get_k_n(g, n)

	#	diagVal = K_nn_dot - 2 * np.dot(np.dot(k_n_dot.T, g['K_M_inv']), k_n)
	#	diagVal += np.dot(np.dot(np.dot(np.dot(k_n.T, g['K_M_inv']), v['K_M_dot']), g['K_M_inv']), k_n)
	#	Gamma_dot[n][n] = diagVal / sigma2
        
        if varIndex == 0:
                K_N_dot_diag = np.ones(N)
        else:
                K_N_dot_diag = np.zeros(N)

        Lambda_diag = K_N_dot_diag - 2 * np.diagonal(np.dot(np.dot(v['K_NM_dot'],g['K_M_inv']),g['K_NM'].T))
        Lambda_diag += np.diagonal(np.dot(np.dot(np.dot(np.dot(g['K_NM'], g['K_M_inv']), v['K_M_dot']), g['K_M_inv']), g['K_NM'].T))
        np.fill_diagonal(Gamma_dot,np.multiply(Lambda_diag,math.pow(sigma2,-1)))
	return Gamma_dot

#0 unless taking derivative wrt c
def get_K_nn_dot(X, c, b, n, varIndex):
	if varIndex == 0:
		return kernel(X[n,:], X[n,:], c, b) / c
	return 0

#This is the transposed n:th row in K_NM_dot
def get_k_n_dot(v, n):
	return (v['K_NM_dot'][n,:]).reshape(((v['K_NM_dot'][n,:]).shape[1], 1))

#This is the transposed n:th row in K_NM
def get_k_n(g, n):
	return (g['K_NM'][n,:]).reshape(((g['K_NM'][n,:]).shape[1], 1))

def get_A_dot(sigma2, g, v):
	A_dot = sigma2 * v['K_M_dot'] + np.dot(v['K_NM_bar_dot'].T, g['K_NM_bar'])
	return A_dot + (np.dot(v['K_NM_bar_dot'].T, g['K_NM_bar'])).T - np.dot(np.dot(g['K_NM_bar'].T, v['Gamma_bar_dot']), g['K_NM_bar'])

#Used for the predictive distribution
def get_k_star(x_star, Xbar, c, b):
	return kernelMatrix(Xbar, x_star, c, b)

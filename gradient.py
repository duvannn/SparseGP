import numpy as np
import scipy as sp
import scipy.optimize as opt
import math
import random
from utils import *
from sys import argv

#Class used for automatic test error evaluation, using sets of different n.o. pseudopoints and training set limits
class testRunner():

	dataset = "kin40k" #"kin40k" or "pumadyn32nm"
	pseudopoints = [5, 20]
	trainingsetLimits = [500, 1000]

	def run(self):
		for i, ppoints in enumerate(self.pseudopoints):
			notCleared = True
			while notCleared:
				try:
					print "Dataset: " + self.dataset + ", pseudo points: " + str(ppoints) + ", training set size limit: " + str(self.trainingsetLimits[i])  
					s = spgp(self.trainingsetLimits[i], self.dataset)
					s.train(ppoints)
					print "Done Training"
					print "c"
					print s.c
					print "sigma2"
					print s.sigma2
					print "b"
					print s.b
					print "Xbar"
					print s.Xbar
					testError = s.evaluateTestError()
					print "test error: " + str(testError)
					notCleared = False
				
				except:
					notCleared = True

#########
# This is an implementation of the SPGP, trained by using the manually derived gradients.
# Usage: 
# from gradient import *
# s = spgp()
# s.train() #Trains the GP hyperparameters (with an ARD kernel) and pseudo inputs on the kin40k or pumadyn32nm dataset 
# s.evaluateTestError() #evaluates MSE on the test set.
##########
#params structure [c, sigma^2, b_1,...,b_D, xbar_1^1,...,xbar_1^D, ..., xbar_M^1,...,xbar_M^D]
class spgp():

	#Boundary values for all parameters
	c_limit = (0.0000000000001, None)
	sigma2_limit = (0.0000000000001, None)
	b_limits = (0.0000000000001, None)
	xbar_limits = (None, None)

	#limit is used to limit n.o training points
	def __init__(self, limit = 30, dataset = "kin40k"):
		trX, trY, tX, tY = get_all_data(dataset)
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

	#Return MSE (average of (prediction_mean_i - test_output_i)^2 for all test examples i)
	#Prediction uses the mean of the predictive distribution p(y_star|x_star, y, X, Xbar)
	def evaluateTestError(self):
		g = generalGradientVars(self.trainX, self.Xbar, self.trainY, self.sigma2, self.c, self.b, self.M, True)
		LambdaAndSigma2_inv_diag = 1 / (g['Gamma_diag'] * self.sigma2)
		Q_M = g['K_M'] + np.dot(g['K_NM'].T, diagDotMatrix(LambdaAndSigma2_inv_diag, g['K_NM']))
		Q_M_inv = np.linalg.inv(Q_M)
		predMatrix = np.dot(Q_M_inv, np.dot(g['K_NM'].T, diagDotMatrix(LambdaAndSigma2_inv_diag, self.trainY)))

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

#Returns the kernel function on two input points only (x_1 and x_2) 
#(actually in this code only used when sending in identical inputs, so the output is always c)
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
	logDet = 0.0
	for elem in g['Gamma_diag']:
		logDet += np.log(elem)

	phi_1 = math.log(np.linalg.det(g['A'])) + logDet - math.log(np.linalg.det(g['K_M'])) 
	phi_1 += (N-M) * math.log(sigma2)

	GammaInv_y = diagDotMatrix(g['Gamma_inv_diag'], y)
	innerMatrix = GammaInv_y - diagDotMatrix(g['Gamma_inv_diag'], np.dot(g['K_NM'], np.dot(g['A_inv'], np.dot(g['K_NM'].T, GammaInv_y))))
	phi_2 = (sigma2 ** (-1)) * np.dot(y.T, innerMatrix)

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

def gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, index):
	v = variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, index)
	pd1 = phiDot_1(g, v)
	pd2 = phiDot_2(sigma2, g, v)
	return 0.5 * (pd1 + pd2)

#phi_1 and phi_2 will be calculated here explicitely since it is of different form than for the other gradients
def gradient_wrt_sigma2(y, sigma2, g):
	sigma2_inv = math.pow(sigma2, -1)

	#phi_1
	term_1 = sigma2_inv * g['Gamma_inv_diag'].sum()
	componentMatrix1 = diagDotMatrix(g['Gamma_inv_diag'], np.dot(g['K_NM'], g['A_inv'])) # NxM
	componentMatrix2 = matrixDotDiag(g['K_NM'].T, g['Gamma_inv_diag']) # MxN
	term_2 = -sigma2_inv * np.trace(np.dot(componentMatrix1, componentMatrix2)) # Might need to do manual (NxN too large?)
	phi_1 = term_1 + term_2

	#phi_2
	GammaInv_y = diagDotMatrix(g['Gamma_inv_diag'], y)
	term_2 = diagDotMatrix(g['Gamma_inv_diag'], np.dot(g['K_NM'], np.dot(g['A_inv'], np.dot(g['K_NM'].T, GammaInv_y))))
	phi_2 = -(sigma2_inv**2) * (np.linalg.norm(GammaInv_y - term_2) ** 2)

	return 0.5 * phi_1 + 0.5 * phi_2

#Derivatives of phi_1 and phi_2, using general info in g and variable specific info in v
#assuming that in derivations A^(T/2) means A^(1/2).T
def phiDot_1(g, v):
	A_term = g['A_half_inv'].dot(v['A_dot'].dot(g['A_half_inv'].T)) 
	K_term = g['K_M_half_inv'].dot(v['K_M_dot'].dot(g['K_M_half_inv'].T))
	return np.trace(A_term) + v['Gamma_bar_dot_diag'].sum() - np.trace(K_term)

def phiDot_2(sigma2, g, v):
	Term1 = - np.dot(g['y_Gamma'].T, diagDotMatrix(v['Gamma_bar_dot_diag'], g['y_Gamma']))
	Term2 = 2 * g['y_Gamma'].T.dot(diagDotMatrix(v['Gamma_bar_dot_diag'], g['K_NM_bar'].dot(g['A_inv'].dot(g['K_NM_bar'].T.dot(g['y_Gamma'])))))
	Term3 = -2 * g['y_Gamma'].T.dot(g['K_NM_bar'].dot(g['A_inv'].dot(v['K_NM_bar_dot'].T.dot(g['y_Gamma']))))
	Term4 = g['y_Gamma'].T.dot(g['K_NM_bar'].dot(g['A_inv'].dot(v['A_dot'].dot(g['A_inv'].dot(g['K_NM_bar'].T.dot(g['y_Gamma']))))))
	return sigma2**(-1) * (Term1 + Term2 + Term3 + Term4)

#Calculations of matrices and vectors used by gradients in a dict
#General: used by all gradients
#limitCalcs: limits calculations done since a subset of the calcs are used in loglikelihood and prediction
#The matrices marked NxN can be very large but diagonal, so to avoid memory issues the diagonal only is used:
#when doing matrix multiplication with these the matrixDotDiag and diagDotMatrix methods are used
#As an imrovement, look into implementing them with sparse matrices
def generalGradientVars(X, Xbar, y, sigma2, c, b, M, limitCalcs = False):
	g = {}

	g['K_M'] = get_K_M(Xbar, c, b)
	g['K_M_inv'] = np.linalg.inv(g['K_M'])
	g['K_NM'] = get_K_NM(X, Xbar, c, b)

	g['Gamma_diag'] = get_Gamma_diag(sigma2, X, Xbar, g['K_M_inv'], c, b) #NxN
	g['Gamma_inv_diag'] = 1 / g['Gamma_diag'] #NxN

	g['A'] = get_A(sigma2, g) 
	g['A_inv'] = np.linalg.inv(g['A'])

	if limitCalcs: return g
	
	g['A_half'] = np.linalg.cholesky(g['A'])
	g['A_half_inv'] = np.linalg.inv(g['A_half'])
	
	g['Gamma_half_diag'] = np.power(g['Gamma_diag'], 0.5) #NxN
	g['Gamma_half_inv_diag'] = 1 / g['Gamma_half_diag'] #NxN

	g['K_M_half'] = np.linalg.cholesky(g['K_M'])		
	g['K_M_half_inv'] = np.linalg.inv(g['K_M_half'])

	g['K_NM_bar'] = diagDotMatrix(g['Gamma_half_inv_diag'], g['K_NM'])
	g['y_Gamma'] = diagDotMatrix(g['Gamma_half_inv_diag'], y)

	return g

#Variable specific: different when taking gradient wrt different variables
#The matrices marked NxN can be very large but diagonal, so to avoid memory issues the diagonal only is used:
#when doing matrix multiplication with these the matrixDotDiag and diagDotMatrix methods are used
#As an imrovement, look into implementing them with sparse matrices
def variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, varIndex):
	v = {}

	v['K_NM_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'NM')
	v['K_NM_bar_dot'] = diagDotMatrix(g['Gamma_half_inv_diag'], v['K_NM_dot'])
	v['K_M_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'M')

	v['Gamma_dot_diag'] = get_Gamma_dot_diag(X, c, sigma2, b, g, v, varIndex) #NxN
	v['Gamma_bar_dot_diag'] = g['Gamma_half_inv_diag'] * v['Gamma_dot_diag'] * g['Gamma_half_inv_diag'] #NxN

	v['A_dot'] = get_A_dot(sigma2, g, v)

	return v

#Matrix and vector calculations used by neg log likelihood and gradient calculations
def get_K_M(Xbar, c, b):
	return kernelMatrix(Xbar,Xbar,c,b)

def get_K_NM(X, Xbar, c, b):
	return kernelMatrix(X,Xbar,c,b)

def get_A(sigma2, g):
	return sigma2 * g['K_M'] + np.dot(matrixDotDiag(g['K_NM'].T, g['Gamma_inv_diag']), g['K_NM'])

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
	return A_dot + (np.dot(v['K_NM_bar_dot'].T, g['K_NM_bar'])).T - np.dot(matrixDotDiag(g['K_NM_bar'].T, v['Gamma_bar_dot_diag']), g['K_NM_bar'])

#Used for the predictive distribution (x_star is the new input point)
def get_k_star(x_star, Xbar, c, b):
	return kernelMatrix(Xbar, x_star, c, b)

#Multiply a matrix with a diagonal matrix whose diagonal is given by diagEntries
#Multiply a matrix by a diag matrix with diagEntries
def matrixDotDiag(matrix, diagEntries):
	retMat = np.asarray(matrix) * diagEntries
	return retMat

#Multiply a diagonal matrix whose diagonal is given by diagEntries with a matrix
#Multiply a diag matrix with diagEntries with matrix
def diagDotMatrix(diagEntries, matrix):
	retMat = np.asarray(matrix) * np.asarray(np.matrix(diagEntries).T)
	return retMat

#Return diagonal entries of Gamma
def get_Gamma_diag(sigma2, X, Xbar, K_M_inv, c, b):
	N = X.shape[0]
	Lambda = np.zeros(N)

	for n in range(0, N):
		x_n = X[n,:]
		K_nn = kernel(x_n, x_n, c, b)
		k_x_n = kernelMatrix(Xbar, x_n, c, b)			
		Lambda[n] =  K_nn - np.dot(np.dot(k_x_n.T, K_M_inv), k_x_n)[0,0]
	return 1 + Lambda / sigma2

#Return diagonal entries of Gamma_dot
#varIndex: the index of the relevant variable in the params vector
def get_Gamma_dot_diag(X, c, sigma2, b, g, v, varIndex):
	N = X.shape[0]
	Gamma_dot = np.zeros(N)

	for n in range(0, N):
		K_nn_dot = get_K_nn_dot(X, c, b, n, varIndex)
		k_n_dot = get_k_n_dot(v, n)
		k_n = get_k_n(g, n)

		diagVal = K_nn_dot - 2 * np.dot(np.dot(k_n_dot.T, g['K_M_inv']), k_n)
		diagVal += np.dot(np.dot(np.dot(np.dot(k_n.T, g['K_M_inv']), v['K_M_dot']), g['K_M_inv']), k_n)
		Gamma_dot[n] = diagVal / sigma2

	return Gamma_dot
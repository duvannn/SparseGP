import numpy as np
import scipy as sp
import scipy.optimize as opt
import math
import random
from utils import *
from sys import argv

#params structure [c, sigma^2, b_1,...,b_D, xbar_1^1,...,xbar_1^D, ..., xbar_M^1,...,xbar_M^D]
class gp():

	#limit is used temporarily to limit n.o training points
	def __init__(self, M = 20, limit = 1000):
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
		#return neg_loglikelihood(params0, self.trainX, self.trainY, self.M)
		return opt.fmin_cg(neg_loglikelihood, params0, fprime=gradient, args=(self.trainX, self.trainY, self.M))

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

	logLik = (0.5 * (phi_1 + phi_2 + N * math.log(2*math.pi)))[0,0]
	print logLik
	return logLik

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
	gradVec = [None for i in range(0, D)]
	for i in range(0,D):
		gradVec[i] = gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, 2+i)
	return gradVec

def gradient_wrt_Xbar(X, Xbar, y, sigma2, c, b, M, g):
	D = X.shape[1]
	gradVec = [None for i in range(0, M*D)]
	for i in range(0, M*D):
		gradVec[i] = gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, 2+D+i)
	return gradVec

#phi_1 and phi_2 will be calculated here explicitely since it is of different form than for the other gradients
def gradient_wrt_sigma2(y, sigma2, g):
	Gamma_inv = np.linalg.inv(g['Gamma'])
	sigma2_inv = math.pow(sigma2, -1)
	Z = np.dot(g['K_NM'], np.dot(np.linalg.inv(g['A']),g['K_NM'].T)) #auxiliery 
	U = y.T.dot(Gamma_inv.dot(Z.dot(np.power(Gamma_inv,2).dot(y)))) #auxiliery 
	phi_1 = sigma2_inv*np.matrix.trace(Gamma_inv) - sigma2_inv * np.matrix.trace(np.dot(Gamma_inv,np.dot(Z,Gamma_inv)))
	phi_2 = -sigma2_inv**2*(np.linalg.norm(np.dot(Gamma_inv,y))**2 + np.linalg.norm(Gamma_inv.dot(np.dot(Z,np.dot(Gamma_inv,y))))**2-U-U.T)
	return 0.5 * phi_1 + 0.5 * phi_2

def gradientForVarIndex(X, Xbar, y, sigma2, c, b, M, g, index):
	v = variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, index)
	pd1 = phiDot_1(g, v)
	pd2 = phiDot_2(sigma2, g, v)
	return 0.5 * (pd1 + pd2)

#Derivatives of phi_1 and phi_2, using general info in g and variable specific info in v
#assuming that in derivations A^(T/2) means A^(1/2).T
def phiDot_1(g, v):
	A_term = np.linalg.inv(g['A_half']).dot(v['A_dot'].dot(np.linalg.inv(g['A_half'].T))) 
	Gamma_term = v['Gamma_bar_dot']
	K_term = np.linalg.inv(g['K_M_half']).dot(v['K_M_dot'].dot(np.linalg.inv(g['K_M_half'].T)))
	return np.trace(A_term) + np.trace(Gamma_term) - np.trace(K_term)

def phiDot_2(sigma2, g, v):
	Term1 = - g['y_Gamma'].T.dot(v['Gamma_bar_dot'].dot(g['y_Gamma']))
	Term2 = 2 * g['y_Gamma'].T.dot(v['Gamma_bar_dot'].dot(g['K_NM_bar'].dot(np.linalg.inv(g['A']).dot(g['K_NM_bar'].T.dot(g['y_Gamma'])))))
	Term3 = -2 * g['y_Gamma'].T.dot(g['K_NM_bar'].dot(np.linalg.inv(g['A']).dot(v['K_NM_bar_dot'].T.dot(g['y_Gamma']))))
	Term4 = g['y_Gamma'].T.dot(g['K_NM_bar'].dot(np.linalg.inv(g['A']).dot(v['A_dot'].dot(np.linalg.inv(g['A']).dot(g['K_NM_bar'].T.dot(g['y_Gamma']))))))
	return sigma2**(-1) * (Term1 + Term2 + Term3 + Term4)

#Calculations of matrices and vectors used by gradients in a dict
#General: used by all gradients
def generalGradientVars(X, Xbar, y, sigma2, c, b, M):
	g = {}

	g['K_M'] = get_K_M(Xbar, c, b)
	g['K_M_inv'] = np.linalg.inv(g['K_M'])
	g['K_NM'] = get_K_NM(X, Xbar, c, b)
	g['Gamma'] = get_Gamma(sigma2, X, Xbar, g['K_M_inv'], c, b)
	g['A'] = get_A(sigma2, g['K_M'], g['K_NM'], g['Gamma'])
	
	g['A_half'] = np.linalg.cholesky(g['A']) 
	g['Gamma_half'] = np.linalg.cholesky(g['Gamma'])
	g['K_M_half'] = np.linalg.cholesky(g['K_M'])

	g['K_NM_bar'] = np.dot(np.linalg.inv(g['Gamma_half']), g['K_NM'])
	g['y_Gamma'] = np.linalg.inv(g['Gamma_half']).dot(y)

	return g

#Variable specific: different when taking gradient wrt different variables
def variableSpecificGradientVars(X, Xbar, y, sigma2, c, b, M, g, varIndex):
	v = {}

	v['K_NM_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'NM')
	v['K_NM_bar_dot'] = np.dot(np.linalg.inv(g['Gamma_half']), v['K_NM_dot'])
	v['K_M_dot'] = get_K_dot(X, Xbar, c, b, g, varIndex, 'M')
	
	v['Gamma_dot'] = get_Gamma_dot(X, c, sigma2, b, g, v, varIndex)
	v['Gamma_bar_dot'] = np.dot(np.dot(np.linalg.inv(g['Gamma_half']), v['Gamma_dot']), np.linalg.inv(g['Gamma_half']))
	v['A_dot'] = get_A_dot(sigma2, g, v)

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
	for n in range(0,N):
		K_nn_dot = get_K_nn_dot(X, c, b, n, varIndex)
		k_n_dot = get_k_n_dot(v, n)
		k_n = get_k_n(g, n)

		diagVal = K_nn_dot - 2 * np.dot(np.dot(k_n_dot.T, g['K_M_inv']), k_n)
		diagVal += np.dot(np.dot(np.dot(np.dot(k_n.T, g['K_M_inv']), v['K_M_dot']), g['K_M_inv']), k_n)
		Gamma_dot[n][n] = diagVal / sigma2

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
	
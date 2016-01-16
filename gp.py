from __future__ import division
from theano import tensor as T
from theano import function
from theano import gradient
from theano.tensor import slinalg
import numpy as np
from scipy.optimize import minimize
from kernels import ARDKernel, get_exp
from theano import shared
from theano import config
from keras import constraints,optimizers
"""
Standard gaussian process implementation with ARD kernel. Hyperparams: sigma, c, b_1...,b_D.
"""

class StrictConstraint(constraints.Constraint):
	def __call__(self, p):
		p *= T.cast(p > 0.0001, config.floatX)
		return p

class GaussianProcess(object):
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.N = self.xtrain.shape[0]
        self.D = self.xtrain.shape[1]
        self.b = shared(np.ones(self.D).astype(config.floatX),"b")
        self.c = shared(numpy.asarray(1., dtype=theano.config.floatX),"c")
        self.sigma = shared(numpy.asarray(1., dtype=theano.config.floatX),"sigma")
        self.kernel = ARDKernel(self.D,b = self.b,c=self.c)
        self.det = T.nlinalg.Det()
        self.inverter = T.nlinalg.MatrixInverse()
        self.params = [self.b,self.c,self.sigma]

    def updateParams(updates):
    	for param, update in zip(self.params,updates):
    		param.set_value(update)

    def eval_kernel(self):
    	return self.kernel.evaluate(self.xtrain, self.xtrain)
        
    def getPosteriorPredictive(self):
    	pred_x = T.dmatrix("pred x")
    	Knn = get_exp(self.xtrain,self.xtrain,self.D,self.b,self.c)
        K_train=Knn + T.pow(self.sigma,2)*T.eye(self.xtrain.shape[0]); 
        K_mix = get_exp(self.xtrain,pred_x,self.D,self.b,self.c)
        K_predict=get_exp(pred_x,pred_x,self.D,self.b,self.c);
        K_k_inv = self.inverter(K_train);
        K_mixinv= T.dot(K_mix.T, K_k_inv);
        mu=T.dot(K_mixinv,self.ytrain);
        sigma_p=K_predict-T.dot(K_mixinv,K_mix);
        return function([pred_x],[mu,sigma_p])

    def sample(self,domain,samples=10):
    	pred = self.getPosteriorPredictive()
        mu, cp = pred(domain)
        return np.random.multivariate_normal(mu.T[0],cp,samples)

    def log_likelihood(self):
    	Knn = get_exp(self.xtrain,self.xtrain,self.D,self.b,self.c)
        c = Knn+ T.pow(self.sigma,2.0) * T.eye(self.N)
        cinv = self.inverter(c)
        t1 = 0.5*T.log(self.det(c))
        t2 = 0.5*T.dot(T.dot(self.ytrain.T,cinv),self.ytrain)
        t3 = 0.5*self.N*T.log(2.0*np.pi);
        tot = t1+t2+t3
        return tot.flatten()[0]

    def getL(self):
        return function(inputs=[],outputs=[self.log_likelihood()])

    def grad(self):
        grad = T.grad(self.log_likelihood(),self.params) 
        fn = function(inputs=[], outputs=grad)
        return fn

    def train_numpy_cost_grads(self, method = "L-BFGS-B"):
        cost = self.getL()
        g = self.grad()
        def train_fn(x):
            b = x[:self.D]
            c = x[self.D]
            sigma = x[self.D+1]
            self.b.set_value(config.floatX(b),borrow=True)
            self.c.set_value(config.floatX(c),borrow=True)
            self.sigma.set_value(config.floatX(sigma),borrow=True)
            c = cost(self.xtrain,self.xtrain)
            grads = g(self.xtrain,self.xtrain)
            gr = list(grads[0])
            gr.append(np.array(grads[1]))
            gr.append(np.array(grads[2]))
            c = config.floatX(c)
            gr = config.floatX(gr)
            return c,gr
        x = np.random.random(self.D+2)
        weights = minimize(train_fn,x,
                    method=method, jac=True,bounds=tuple((0.0001,None) for x in range(self.D+2)),
                    options={'maxiter': 200, 'disp': True})
        x = weights.x
        b = x[:self.D]
        c = x[self.D]
        sigma = x[self.D+1]
        self.b.set_value(config.floatX(b),borrow=True)
        self.c.set_value(config.floatX(c),borrow=True)
        self.sigma.set_value(config.floatX(sigma),borrow=True)
        return weights

    def Adam(self,lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        constraint = [constraints.NonNeg() for i in self.params]
        rmsprop = optimizers.Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
        train_loss = self.log_likelihood()
        updates = rmsprop.get_updates(self.params,constraint,train_loss)
        trainf = function([],[train_loss], updates=updates)
        return trainf

    def RMSprop(self,lr=0.001, rho=0.9, epsilon=1e-6):
        constraint = [constraints.NonNeg() for i in self.params]
        rmsprop = optimizers.RMSprop(lr=lr,tho=rho,epsilon=epsilon)
        train_loss = self.log_likelihood()
        updates = rmsprop.get_updates(self.params,constraint,train_loss)
        trainf = function([],[train_loss], updates=updates)
        return trainf

"""
Sparse Gaussian Processes using Pseudo-inputs, Snelson et al. 2006
Hyperparams: sigma, c, b_1...,b_D , X_11, ... X_1D ... X_MD
"""
class SparseGaussianProcess(GaussianProcess):
	def __init__(self,xtrain,ytrain,M):
		GaussianProcess.__init__(self,xtrain,ytrain)
		self.M = M
		self.pseudo_points = shared(self.xtrain[:self.M,:].astype(config.floatX),"pseudo_inputs")
		self.Kmm = get_exp(self.pseudo_points,self.pseudo_points,self.D,self.b,self.c)
		self.Kmn = get_exp(self.pseudo_points,self.xtrain,self.D,self.b,self.c)
		self.params = [self.b,self.c,self.sigma, self.pseudo_points]
		self.xtrain_batch = T.matrix("xtrain batch")
		self.ytrain_batch = T.matrix("ytrain batch")

	# def log_likelihood(self):
	# 	lamda = T.diag(T.diag(self.c*T.eye(self.N) - T.dot(T.dot(self.Kmn.T,self.inverter(self.Kmm)),self.Kmn)))
	# 	gamma = T.power(self.sigma,-2.)*lamda + T.eye(self.N)
	# 	gammainv = self.inverter(gamma)
	#  	A = T.power(self.sigma,2.)*self.Kmm + T.dot(T.dot(self.Kmn,gammainv),self.Kmn.T);
	#  	A_sholesky = T.slinalg.cholesky(A)
	#  	psi1 = T.log(T._tensor_py_operators.prod(T.diag(A_sholesky)) ) + T.log(self.det(gamma))-T.log(self.det(self.Kmm))+(self.N-self.M)*T.log(T.power(self.sigma,2.))
	#  	y_gamma = T.dot(T.sqrt(gammainv),self.ytrain)
	#  	psi2 = T.dot(T.dot(T.dot(self.inverter(A_sholesky),self.Kmn),T.sqrt(gammainv)),y_gamma)
	#  	psi2 = T.power(self.sigma,-2.)*(T._tensor_py_operators.norm(y_gamma,2.)-T._tensor_py_operators.norm(psi2,2.))
	#  	return (psi1+psi2).flatten()[0]

	def log_likelihood(self):
	 	lamda = T.diag(T.diag(self.c*T.eye(self.N) - T.dot(T.dot(self.Kmn.T,self.inverter(self.Kmm)),self.Kmn)))
	 	gamma = T.power(self.sigma,-2.)*lamda + T.eye(self.N)
	 	gammainv = self.inverter(gamma)
	 	A = T.power(self.sigma,2.)*self.Kmm + T.dot(T.dot(self.Kmn,gammainv),self.Kmn.T);
	 	psi1 = T.log(self.det(A)) + T.log(self.det(gamma))-T.log(self.det(self.Kmm))+(self.N-self.M)*T.log(T.power(self.sigma,2.))
	 	Ainv = self.inverter(A)
	 	psi21 = T.power(1./self.sigma,2.)*self.ytrain.T
	 	psi22 = gammainv-T.dot(T.dot(T.dot(T.dot(gammainv,self.Kmn.T), Ainv),self.Kmn),gammainv)
	 	psi2 = T.dot(T.dot(psi21,psi22),self.ytrain)
	 	return 0.5*(psi1+psi2).flatten()[0]

	def log_likelihood_batch(self):
		Kmn = get_exp(self.pseudo_points,self.xtrain_batch,self.D,self.b,self.c)
		gamma = (T.power(self.sigma,-2.)*self.c + 1)*T.eye(self.N)
		gammainv = (T.power(self.sigma,2.)*T.power(self.c,-1.)+1)*T.eye(self.N)
	 	A = T.power(self.sigma,2.)*self.Kmm + T.dot(T.dot(Kmn,gammainv),Kmn.T);
	 	psi1 = T.log(self.det(A)) + T.log(self.det(gamma))-T.log(self.det(self.Kmm))+(self.N-self.M)*T.log(T.power(self.sigma,2.))
	 	Ainv = self.inverter(A)
	 	psi21 = T.power(1./self.sigma,2.)*self.ytrain_batch.T
	 	psi22 = gammainv-T.dot(T.dot(T.dot(T.dot(gammainv,Kmn.T), Ainv),Kmn),gammainv)
	 	psi2 = T.dot(T.dot(psi21,psi22),self.ytrain_batch)
	 	return 0.5*(psi1+psi2).flatten()[0]




	def getL(self):
		return function(inputs=[],outputs=[self.log_likelihood()])

	def grad(self):
		grad = T.grad(self.log_likelihood(),self.params)
		fn = function(inputs=[], outputs=grad)
		return fn

	def getParams(self,only_psudo = False):
		if only_psudo:
			constraint = [constraints.Constraint()]
			return constraint, [self.pseudo_points]
		else:
			constraint = [StrictConstraint() for i in [self.b,self.c,self.sigma]]
			constraint.append(constraints.Constraint())
			return constraint, self.params

	def RMSprop(self,batch_training=False, only_psudo = False, lr=0.001, rho=0.9, epsilon=1e-6): 
		constraints, params = self.getParams(only_psudo)
		rmsprop = optimizers.RMSprop(lr=lr,tho=rho,epsilon=epsilon)
		if batch_training:
			train_loss = self.log_likelihood_batch()
			updates = rmsprop.get_updates(self.params,constraints,train_loss)
			trainf = function([self.xtrain_batch,self.ytrain_batch],[train_loss], updates=updates)
		else:
			train_loss = self.log_likelihood()
			updates = rmsprop.get_updates(self.params,constraints,train_loss)
			trainf = function([],[train_loss], updates=updates)
		return trainf



	def Adam(self,only_psudo = False, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
		constraints, params = self.getParams(only_psudo)
		adam = optimizers.Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
		train_loss = self.log_likelihood()
		updates = adam.get_updates(self.params,constraints,train_loss)
		trainf = function([],[train_loss], updates=updates)
		return trainf

	def getPosteriorPredictive(self):
		pred_x = T.dmatrix("pred_x")
		gamma = (self.c+ T.power(self.sigma,2.))*T.eye(self.N)
		gammainv = (T.power(self.c,-1.)*np.ones((self.N,1))+ T.power(self.sigma,-2.))*T.eye(self.N)
		K_star_star = get_exp(pred_x,pred_x,self.D,self.b,self.c)
		K_star_m = get_exp(pred_x,self.pseudo_points,self.D,self.b,self.c)
		K_x_xp =  self.Kmn
		Qm = self.Kmm + T.dot(T.dot(K_x_xp,gammainv),K_x_xp.T)
		Qm_inv = self.inverter(Qm)
		mu1 = T.dot(K_star_m,Qm_inv)
		mu2 = T.dot(K_x_xp,gammainv)
		mu = T.dot(T.dot(mu1,mu2),self.ytrain)
		sigmainv = self.inverter(self.Kmm)-Qm_inv
		sigma1 = T.dot(T.dot(K_star_m,sigmainv),K_star_m.T)
		sigma = K_star_star - sigma1 + T.power(self.sigma,2.)*T.eye(pred_x.shape[0])
		return function([pred_x],[mu,sigma])

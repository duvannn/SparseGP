from __future__ import division
from theano import tensor as T
from theano import function
from theano import gradient
from theano.tensor import slinalg
import numpy as np
from scipy.optimize import minimize
from kernels import *
from theano import shared
from keras import constraints
from keras import optimizers
"""
Standard gaussian process implementation with ARD kernel. Hyperparams: sigma, c, b_1...,b_D.
"""
class GaussianProcess(object):
    def __init__(self, xtrain, ytrain):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.N = self.xtrain.shape[0]
        self.D = self.xtrain.shape[1]
        self.b = shared(np.ones(self.D),"b")
        self.c = shared(1.0,"c")
        self.sigma = shared(np.float64(1.),"sigma")
        self.kernel = ARDKernel(self.D,b = self.b,c=self.c)
        self.det = T.nlinalg.Det()
        self.inverter = T.nlinalg.MatrixInverse()
        self.params = [self.b,self.c,self.sigma]

    def updateParams(updates):
    	for param, update in zip(self.params,updates):
    		param.set_value(update)

    def eval_kernel(self):
    	return self.kernel.evaluate(self.xtrain, self.xtrain)
        
    def getPosteriorPredictive(self,pred_x):
        K_train=self.kernel.evaluate(self.xtrain,self.xtrain) + T.pow(self.sigma,2)*T.eye(self.xtrain.shape[0]); 
        K_mix=self.kernel.evaluate(self.xtrain,pred_x); 
        K_predict=self.kernel.evaluate(pred_x,pred_x);
        K_k_inv = self.inverter(K_train);
        K_mixinv= T.dot(K_mix.T, K_k_inv);
        mu=T.dot(K_mixinv,self.ytrain);
        sigma_p=K_predict-T.dot(K_mixinv,K_mix);
        return mu.eval(), sigma_p.eval()

    def sample(self,domain,samples=10):
        mu, cp = self.getPosteriorPredictive(domain)
        return np.random.multivariate_normal(mu.T[0],cp,samples)

    def log_likelihood(self):
        c = self.kernel.exp + T.pow(self.sigma,2.0) * T.eye(self.N)
        cinv = self.inverter(c)
        t1 = 0.5*T.log(self.det(c))
        t2 = 0.5*T.dot(T.dot(self.ytrain.T,cinv),self.ytrain)
        t3 = 0.5*self.N*T.log(2.0*np.pi);
        tot = t1+t2+t3
        return tot.flatten()[0]


	def np_log_likelihood(self):
		c = self.kernel.evaluate(self.xtrain,self.xtrain) + np.power(sigma,2)*np.eye(self.xtrain.shape[0])
		t1 = 0.5*np.log(np.linalg.det(c))
		t2 = 0.5*np.dot(np.dot(self.ytrain.T,self.np.linalg.inv(c)),self.ytrain)
		t3 = 0.5*self.N*np.log(2.0*np.pi)
		return t1+t2+t3

    def getL(self):
        return function(inputs=[self.kernel.X, self.kernel.Xp],outputs=[self.log_likelihood()])

    def grad(self):
        grad = T.grad(self.log_likelihood(),self.params) #+T.log(self.c)+T.sum(T.log(self.b))+T.log(self.sigma)
        fn = function(inputs=[self.kernel.X,self.kernel.Xp], outputs=grad)
        return fn

    def plain_gradient_descent(self,maxiters =1000, step=0.01):
        gf = self.grad()
        L = self.getL()
        costs = []
        iters = 0
        prevcost = np.infty
        currcost = L(self.xtrain,self.xtrain)[0]
        costs.append(currcost)
        while iters<maxiters:
            updates = gf(self.xtrain,self.xtrain)
            costs.append(L(self.xtrain,self.xtrain)[0])
            for update,param in zip(updates,self.params):
            	param.set_value(np.float64(param.get_value()-step * np.clip(update,-0.1,0.1)))
            iters+=1
            prevcost = costs[iters-1]
            currcost = costs[iters]
        return costs

    def train_numpy_cost_grads(self, method = "L-BFGS-B"):
        cost = self.getL()
        g = self.grad()
        def train_fn(x):
            b = x[:self.D]
            c = x[self.D]
            sigma = x[self.D+1]
            self.b.set_value(np.float64(b),borrow=True)
            self.c.set_value(np.float64(c),borrow=True)
            self.sigma.set_value(np.float64(sigma),borrow=True)
            c = cost(self.xtrain,self.xtrain)
            grads = g(self.xtrain,self.xtrain)
            gr = list(grads[0])
            gr.append(np.array(grads[1]))
            gr.append(np.array(grads[2]))
            c = np.float64(c)
            gr = np.float64(gr)
            return c,gr
        x = np.random.random(self.D+2)
        weights = minimize(train_fn,x,
                    method=method, jac=True,bounds=tuple((0.0001,None) for x in range(self.D+2)),
                    options={'maxiter': 200, 'disp': True})
        x = weights.x
        b = x[:self.D]
        c = x[self.D]
        sigma = x[self.D+1]
        self.b.set_value(np.float64(b),borrow=True)
        self.c.set_value(np.float64(c),borrow=True)
        self.sigma.set_value(np.float64(sigma),borrow=True)
        return weights

    def Adam(self,lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        constraint = [constraints.NonNeg() for i in self.params]
        rmsprop = optimizers.Adam(lr=lr,beta_1=beta_1,beta_2=beta_2,epsilon=epsilon)
        train_loss = self.log_likelihood()
        updates = rmsprop.get_updates(self.params,constraint,train_loss)
        trainf = function([self.kernel.X,self.kernel.Xp],[train_loss], updates=updates)
        return trainf

    def RMSprop(self,lr=0.001, rho=0.9, epsilon=1e-6):
        constraint = [constraints.NonNeg() for i in self.params]
        rmsprop = optimizers.RMSprop(lr=lr,tho=rho,epsilon=epsilon)
        train_loss = self.log_likelihood()
        updates = rmsprop.get_updates(self.params,constraint,train_loss)
        trainf = function([self.kernel.X,self.kernel.Xp],[train_loss], updates=updates)
        return trainf

"""
Sparse Gaussian Processes using Pseudo-inputs, Snelson et al. 2006
HYperparams: sigma, c, b_1...,b_D , X_11, ... X_1D ... X_MD
"""
class SparseGaussianProcess(GaussianProcess):
	def __init__(self,xtrain,ytrain,M):
		GaussianProcess.__init__(self,xtrain,ytrain)
		self.M = M
		self.pseudo_points = shared(self.xtrain[:self.M,:],"pseudo_inputs")
		self.Kmm = ARDKernel(self.D,b=self.b,c=self.c)
		self.Kmn = ARDKernel(self.D,b = self.b,c = self.c)
		self.Knm = ARDKernel(self.D,b = self.b,c=self.c)
		self.params = [self.b,self.c,self.sigma, self.pseudo_points]

	def log_likelihood(self):
	 	gamma = T.diag(self.kernel.exp) + T.power(self.sigma,2.)*T.eye(self.N)
		gammainv = self.inverter(gamma)
	 	Kmm = self.Kmm.exp
	 	Knm = self.Knm.exp
	 	Kmn = self.Kmn.exp
	 	A = T.power(self.sigma,2)*Kmm + T.dot(T.dot(Knm,gammainv),Kmn);
	 	Ainv = self.inverter(A)
	 	psi1 = T.log(self.det(A)) + T.log(self.det(gamma))-T.log(self.det(Kmm))+(self.N-self.M)*T.log(T.power(self.sigma,2))
	 	psi2 = T.dot(T.dot(T.power(1./self.sigma,2)*self.ytrain.T,gammainv-T.dot(T.dot(T.dot(T.dot(gammainv,Knm),Ainv),Kmn),gammainv)),self.ytrain)
	 	return (psi1+psi2).flatten()[0]

	def getL(self):
		return function(inputs=[self.kernel.X, self.kernel.Xp,self.Kmm.X,self.Kmm.Xp,self.Kmn.X,self.Kmn.Xp,self.Knm.X,self.Knm.Xp],outputs=[self.log_likelihood()])

	def grad(self):
		grad = T.grad(self.log_likelihood(),self.params)
		fn = function(inputs=[self.kernel.X, self.kernel.Xp,self.Kmm.X,self.Kmm.Xp,self.Kmn.X,self.Kmn.Xp,self.Knm.X,self.Knm.Xp], outputs=grad)
		return fn


	def plain_gradient_descent(self,maxiters =1000, step=0.01):
		gf = self.grad()
		L = self.getL()
		costs = []
		iters = 0
		prevcost = np.infty
		currcost = L(self.xtrain,self.xtrain,self.pseudo_points.eval(),self.pseudo_points.eval(),
			self.pseudo_points.eval(),self.xtrain,self.xtrain,self.pseudo_points.eval())[0]
		costs.append(currcost)
		while iters<maxiters:
		    updates = gf(self.xtrain,self.xtrain)
		    costs.append(L(self.xtrain,self.xtrain)[0])
		    for update,param in zip(updates,self.params):
		    	param.set_value(np.float64(param.get_value()-step * np.clip(update,-0.1,0.1)))
		    iters+=1
		    prevcost = costs[iters-1]
		    currcost = costs[iters]
		return costs

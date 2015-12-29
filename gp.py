from __future__ import division
from theano import tensor as T
from theano import function
import numpy as np
from scipy.optimize import minimize
from kernels import *
from theano import shared

class GaussianProcess(object):
    def __init__(self, xtrain, ytrain,kernel=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.nuv = shared(np.ones((xtrain.shape[1])).astype(np.float32))
        if(kernel == None):
            self.kernel = ARDKernel()
        else:
            self.kernel = kernel
        self.nu = self.kernel.nu
        self.theta = self.kernel.theta
        self.N = self.xtrain.shape[0]
        self.theta = 1.0
        self.inverter = T.nlinalg.MatrixInverse()
        self.det = T.nlinalg.Det()

    def eval_kernel(self):
    	return self.kernel.evaluate(self.xtrain, self.xtrain,self.nuv.eval(),self.theta)
        
    def getPosteriorPredictive(self,pred_x = np.matrix(np.linspace(-2,2,100,dtype=np.float32)).T):
        K_train=self.kernel.evaluate(self.xtrain,self.xtrain,self.nuv.eval(),self.theta) + T.pow(self.theta,2)*T.eye(self.xtrain.shape[0]); 
        K_mix=self.kernel.evaluate(self.xtrain,pred_x,self.nuv.eval(),self.theta); 
        K_predict=self.kernel.evaluate(pred_x,pred_x,self.nuv.eval(),self.theta);
        K_k_inv = self.inverter(K_train);
        K_mixinv= T.dot(K_mix.T, K_k_inv);
        mu=T.dot(K_mixinv,self.ytrain);
        sigma_p=K_predict-T.dot(K_mixinv,K_mix);
        return mu, sigma_p
    
    def sample(self,samples=10, domain = np.matrix(np.linspace(-2,2,100,dtype=np.float32)).T):
        mu, cp = self.getPosteriorPredictive(domain)
        return np.random.multivariate_normal(mu.T.eval()[0],cp.eval(),samples)
    
    def log_likelihood(self):
        c = self.kernel.exp + T.pow(self.kernel.theta,2) * T.eye(self.kernel.exp.shape[0])
        t1 = 0.5*T.log(self.det(c))
        t2 = 0.5*T.dot(T.dot(self.ytrain.T,self.inverter(c)),self.ytrain)
        t3 = 0.5*self.N*T.log(2.0*np.pi);
        tot = t1+t2+t3
        return tot[0][0]

    def getL(self):
        return function(inputs=[self.kernel.X, self.kernel.Y,self.kernel.nu,self.kernel.theta],outputs=[self.log_likelihood()])

    def grad(self):
        grad = T.grad(self.log_likelihood(),self.kernel.nu)
        fn = function(inputs=[self.kernel.X,self.kernel.Y,self.nu,self.kernel.theta], outputs=[grad])
        return fn

    def plain_gradient_descent(self,maxiters =1000, step=0.01, tresh = 0.001, stop_early = True):
        params= self.nuv.eval()
        gf = self.grad()
        L = self.getL()
        costs = []
        param = []
        iters = 0
        prevcost = np.infty
        currcost = L(self.xtrain,self.xtrain,params,self.theta)[0]
        costs.append(currcost)
        while iters<maxiters or (abs(prevcost-currcost)<tresh and stop_early):
            g = gf(self.xtrain,self.xtrain,params,self.theta)[0]
            costs.append(L(self.xtrain,self.xtrain,params,self.theta)[0])
            #print params
            param.append(params)
            params -= step * g
            iters+=1
            prevcost = costs[iters-1]
            currcost = costs[iters]
        self.nuv.set_value(params)
        return costs, params

    def train_numpy_cost_grads(self, variables = []):
        cost = self.log_likelihood()
        grad = T.grad(cost,self.kernel.nu)
        fn = function(inputs=[self.kernel.X,self.kernel.Y,self.nu,self.kernel.theta], outputs=[cost, grad], allow_input_downcast=True)
        def train_fn(nu):
            c, g = fn(self.xtrain,self.xtrain,np.float32(nu),np.float32(self.theta))
            c = np.float64(c)
            g = np.float64(g)
            return c,g
        x = self.nuv.eval()
        weights = minimize(train_fn,x,
                    method='L-BFGS-B', jac=True,
                    options={'maxiter': 200, 'disp': True})
        nw = weights.x
        nw = np.asarray(nw,dtype=np.float32)
        self.nuv.set_value(nw)
        return weights
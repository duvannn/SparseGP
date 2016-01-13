from __future__ import division
from theano import tensor as T
from theano import function
from theano import shared
import numpy as np

"""
ARD kernel, hyperparameters:c, b_1 .. b_D 
"""
class ARDKernel(object):
    def __init__(self,D,b = None,c = None):
    	if(b==None):
    		self.b = shared(np.ones(D).astype(np.float64),"b")
    	else:
    		self.b = b
    	if(c==None):
        	self.c = shared(np.float64(1.),"c")
        else:
        	self.c = c
        self.X = T.dmatrix('X')
        self.Xp = T.dmatrix('Xp')
        scaledX = T.sqrt(self.b)*self.X
        scaledXp = T.sqrt(self.b)*self.Xp
        squared_euclidean_distances = (scaledX ** 2).sum(1).reshape((scaledX.shape[0], 1)) + (scaledXp ** 2).sum(1).reshape((1, scaledXp.shape[0])) - 2 * scaledX.dot(scaledXp.T)
        xp = 0.5*squared_euclidean_distances
        self.exp = self.c*T.exp(-xp)
        self.C = function(inputs=[self.X, self.Xp], outputs=[self.exp])
    def evaluate(self,X,Y):
        return self.C(X,Y)[0]

def get_exp(X,Xp,D,b,c):
    scaledX = T.sqrt(b)*X
    scaledXp = T.sqrt(b)*Xp
    squared_euclidean_distances = (scaledX ** 2).sum(1).reshape((scaledX.shape[0], 1)) + (scaledXp ** 2).sum(1).reshape((1, scaledXp.shape[0])) - 2 * scaledX.dot(scaledXp.T)
    xp = 0.5*squared_euclidean_distances
    return  c*T.exp(-xp)
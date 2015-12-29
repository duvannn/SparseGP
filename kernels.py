from __future__ import division
from theano import tensor as T
from theano import function

class ARDKernel(object):
    def __init__(self):
        self.theta = T.fscalar("theta")
        self.nu = T.fvector("nu")
        self.X = T.fmatrix('X')
        self.Y = T.fmatrix('Y')
        scaledX = T.sqrt(self.nu)*self.X
        scaledY = self.Y;
        squared_euclidean_distances = ((scaledX.reshape((scaledX.shape[0], 1, -1)) - scaledY.reshape((1, scaledY.shape[0], -1)))**2).sum(2)
        xp = 0.5*squared_euclidean_distances
        self.exp = self.theta*T.exp(-xp)
        self.C = function(inputs=[self.X, self.Y,self.nu,self.theta], outputs=[self.exp])
    def evaluate(self,X,Y,nu,theta):
        return self.C(X,Y,nu,theta)[0]
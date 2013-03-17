import numpy as np
from sklearn.base import BaseEstimator

class IdentityTransformer(object):
	def __init__(self):
		pass
	def transform(self, y):
		return y
	def r_transform(self, ry):
		return ry

class LogTransformer(object):
	def __init__(self):
		pass
	def transform(self, y):
		return np.log(y)
	def r_transform(self, ry):
		return np.exp(ry)
		
class ToDenseMatrix(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        print "DEBUG!!!!!!!!!!!!!!", X.shape
        return X.toarray()
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
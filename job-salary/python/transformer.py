import numpy as np

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
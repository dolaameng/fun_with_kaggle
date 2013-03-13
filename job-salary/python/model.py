from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
import numpy as np
from sklearn import cross_validation
from sklearn import ensemble



def fit_model(model, X, y, features):
	try:
		sub_X = X[:, features]
	except:
		sub_X = X.tocsc()[:, features]
	return model.fit(sub_X, y)
def predict_model(X, model, features):
	try:
		sub_X = X[:, features]
	except:
		sub_X = X.tocsc()[features]
	return model.predict(sub_X)
class RandomFeatureEnsemble(BaseEstimator):
	def __init__(self, ensemble, n_features = 200):
		"""
		ensemble: a list of untrained estimatros
		n_features: number of features used in each forest model
		"""
		self.n_estimators = len(ensemble)
		self.n_features = n_features
		self.feature_sets = []
		self.ensemble = ensemble
	def fit(self, X, y):
		total_rows, total_features = X.shape
		## randomly select features
		bt = cross_validation.Bootstrap(total_features, n_iter = self.n_estimators, 
										train_size = self.n_features)
		self.feature_sets = [fset for (fset, _) in bt]
		"""
		self.ensemble = Parallel(n_jobs = -1)(delayed(fit_model)(self.ensemble[i], X, y, self.feature_sets[i]) 
					for i in xrange(self.n_estimators))
		"""
		self.ensemble = [fit_model(self.ensemble[i], X, y, self.feature_sets[i]) 
					for i in xrange(self.n_estimators)]
		return self
	def predict(self, X):
		"""
		results = Parallel(n_jobs = -1)(delayed(predict_model)(X, self.ensemble[i], self.feature_sets[i]) 
					for i in xrange(self.n_estimators))
		"""
		results = [predict_model(X, self.ensemble[i], self.feature_sets[i]) 
					for i in xrange(self.n_estimators)]
		averaged = np.mean(np.vstack(results), axis = 0)
		return averaged
	def fit_predict(self, X, y):
		return self.fit(X, y).predict(X)
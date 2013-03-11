## model for contract_time - very useful for prediction of salary, but with a lot of missing values
## CURRENT STRATEGY: code missing values as a seperate class MISSING
## use one-hot encoding

from sklearn.base import BaseEstimator
from sklearn import preprocessing
from sklearn import feature_extraction
import numpy as np

class ContractTimeFeatureExtractor(BaseEstimator):
    def __init__(self, na_string = "nan", na_class = 'miss', feature_name='ContractTime'):
        self.na_string = na_string
        self.na_class = na_class
        self.feature_name = feature_name
    def fit(self, X, y=None):
        ## nothing to fit
        return self
    def transform(self, X):
        """ X is a list/np.array of strings
        """
        X = np.asarray(X)
        X[X==self.na_string] = self.na_class
        X = [{self.feature_name:v} for v in X]
        dicter = feature_extraction.DictVectorizer()
        X = dicter.fit_transform(X)
        self.feature_names_ = dicter.get_feature_names()
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)
## build feature_extractor model
from sklearn.base import BaseEstimator

class SalaryFeatureExtractor(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        pass
    def transform(self, X):
        pass
    def fit_transform(self, X, y=None):
        pass
        
class TitleFeatureExtractor(BaseEstimator):
    def __init__(self):
        self.counter = text.CountVectorizer(stop_words = 'english', 
                        ngram_range = (1, 2), binary = True, lowercase = True)
    def fit(self, X, y=None):
        self.counter.fit(X)
        CounterX = self.counter.transform(X)
    def transform(self, X):
        pass
    def fit_transform(self, X, y=None):
        pass
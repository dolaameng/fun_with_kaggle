## generate feature selection model based on pearson or spearman correlation
## with the target (e.g. log(salary))

from sklearn.base import BaseEstimator
from scipy import stats
import numpy as np

class CorrelationFeatureSelector(BaseEstimator):
    def __init__(self, max_features=None, pvalue_threshold =None, metric='pearson'):
        """if max_features is not None, select best features up to it
        else, select features with pvalue >= pvalue_threshold
        metric = {'pearson', 'spearman'}, 'spearman' is slower
        """
        self.max_features = max_features
        self.pvalue_threshold = pvalue_threshold
        self.feature_indices_ = None
        self.metric = stats.pearsonr if metric is 'pearson' else stats.spearmanr    
    def fit(self, X, y=None):
        XX = X.tocsc()
        y = y.reshape(-1, 1)
        n_cols = XX.shape[1]
        significance = [self.metric(y, XX.getcol(i).toarray()) for i in xrange(n_cols)] 
        feat_coeffs = map(lambda t: t[0], significance)
        feat_pvalues = map(lambda t: t[1], significance)
        sorted_index_pvalues = sorted(zip(range(n_cols), feat_pvalues), reverse=True, key=lambda (i,p): p)
        if self.max_features:
            self.feature_indices_ = [i for (i,p) in sorted_index_pvalues[self.max_features]]
        else:
            self.feature_indices_ = [i for (i,p) in sorted_index_pvalues if p >= self.pvalue_threshold]
        return self
    def transform(self, X):
        XX = X.tocsc()[:, self.feature_indices_].tocsr()
        return XX
    def fit_transform(self, X):
        return self.fit(X).transform(X)

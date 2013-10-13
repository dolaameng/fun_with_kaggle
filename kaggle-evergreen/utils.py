import pylab as pl 
import pandas as pd 
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import safe_sparse_dot
from scipy.sparse import issparse

def barplot_discrete_variable(df, col_name, by = None, topn=None, figsize = (14, 3)):
	if by is None:
		value_counts = [(value, subdf.shape[0]) 
						for value, subdf in df.groupby(col_name)]
		if topn is not None:
			value_counts = sorted(value_counts, key=lambda x: x[1], reverse=True)[:topn]
		unique_values, counts = zip(*value_counts)
		pl.figure(figsize=figsize)
		pl.bar(range(len(unique_values)), counts)
		pl.xticks(range(len(unique_values)), unique_values)
		locs, labels = pl.xticks()
		_ = pl.setp(labels, rotation = 45)
		pl.title(col_name)
	else:
		nrows, ncols = len(df[by].unique()), 1
		fig, axes = pl.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
		fig.suptitle(col_name + '|' + by, y = 0.9)
		#fig.subplots_adjust(top = 2)

		
		for i, (by_value, by_df) in enumerate(df.groupby(by)):
			value_counts = [(value, subdf.shape[0]) 
						for value, subdf in by_df.groupby(col_name)]
			if topn is not None:
				value_counts = sorted(value_counts, key=lambda x: x[1], reverse=True)[:topn]
			unique_values, counts = zip(*value_counts)
			axes[i].bar(range(len(unique_values)), counts)
			axes[i].set_xticks(range(len(unique_values)))
			axes[i].set_xticklabels(unique_values)
			labels = axes[i].get_xticklabels()
			_ = pl.setp(labels, rotation = 45)
			axes[i].set_title(by + '=' + str(by_value))

def boxplot_continuous_variable(df, col_name, qs = None, by = None, figsize=(14, 3)):
	qs = qs or [0, 100]
	pl.figure(figsize=figsize)
	q1, q2 = np.percentile(df[col_name].dropna(), qs)
	df[(q1 <= df[col_name]) *(df[col_name]<= q2)].boxplot(col_name, by = by)

def plot_feature_importances(tree_ensemble, feature_names, figsize = (30, 12)):
	"""
	coursety of https://github.com/ogrisel/notebooks
	"""
	feature_importances = np.mean([t.tree_.compute_feature_importances(normalize=True) 
						for t in tree_ensemble.estimators_], axis = 0)
	pl.figure(figsize = figsize)
	pl.bar(range(feature_importances.shape[0]), feature_importances)
	pl.xticks(range(feature_importances.shape[0]), feature_names)
	ticks, labels = pl.xticks()
	pl.setp(labels, rotation = 90, fontsize=20)
	pl.title('feature importance')
	return sorted(zip(feature_names, feature_importances), 
				key = lambda x: x[1],
				reverse = True)

class SparseToDense(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		return X.toarray()

def soft_threshold(data, centers, thr = 'median', normalized = False):
	"""
	coursety of http://nbviewer.ipython.org/4403811/MNIST%20non%20linear%20feature%20expansion%20for%20classification.ipynb
	"""
	X = normalize(data)
	C = normalize(centers)
	S = safe_sparse_dot(X, C.T)
	if issparse(S):
		S = S.toarray()
	## remove 50% of coding
	if thr is 'median':
		thr = np.median(S)
	elif thr is 'mean':
		thr = S.mean()
	else:
		thr = thr 
	S[S < thr] = 0.0
	if normalized:
		return normalize(S, copy = True)
	else:
		return S
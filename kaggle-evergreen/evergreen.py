import pandas as pd 
import json 
import numpy as np 
from utils import soft_threshold
from urlparse import urlsplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn import metrics
from collections import Counter, defaultdict

def read_data(fname):
	data = pd.read_csv(fname, sep='\t', na_values=['?', 'null'])
	## nontext
	data['domain'] = data['url'].apply(lambda x: '.'.join(urlsplit(x).netloc.split('.')[0:]))
	data['embed_ratio'].loc[data['embed_ratio'] < 0] = np.nan 
	data.loc[pd.isnull(data.alchemy_category), 'alchemy_category'] = 'unknown'
	domain_counts = sorted([(domain, df.shape[0]) for domain, df in data.groupby('domain')],
                       key = lambda x: x[1], reverse = True)
	domain_counts = dict(domain_counts)
	data['rough_domain'] = data['domain'].apply(lambda x: x if domain_counts[x] > 30 else 'others')
	## text - title and body
	data['boilerplate'] = data['boilerplate'].apply(json.loads)
	data['title'] = data['boilerplate'].apply(lambda x: x.get('title', ''))
	data['body'] = data['boilerplate'].apply(lambda x: x.get('body', ''))
	data.loc[pd.isnull(data.title), 'title'] = ''
	data.loc[pd.isnull(data.body), 'body'] = ''
	data['text'] = data.apply(lambda r: ' '.join(r.loc[['title', 'body']]), axis = 1)
	## useful features based on exploration
	useful_features = [u'linkwordscore', u'frameTagRatio', u'non_markup_alphanum_characters', 
	u'numwords_in_url', u'spelling_errors_ratio', u'commonlinkratio_1', 
	'avglinksize', u'numberOfLinks', u'parametrizedLinkRatio', u'html_ratio', 
	u'commonlinkratio_2', u'commonlinkratio_3', u'compression_ratio', 
	u'commonlinkratio_4', u'image_ratio', 'alchemy_category', 'text']
	if 'label' in data.columns:
		return data[useful_features+['label']]
	else:
		return data[useful_features]

class AlchemyDiscretizer(BaseEstimator, TransformerMixin):
	def __init__(self, useful_features = None):
		useful_features = None or ['recreation', 'business', 'sports', 'unknown', 'arts_entertainment', 'computer_internet', 'health', 'culture_politics', 'science_technology', 'religion', 'gaming', 'law_crime', 'weather']
		self.useful_features = useful_features
		self.alchemy_category_labeler_ = LabelBinarizer()
	def fit(self, X, y = None):
		"""X: pd.DataFrame
		"""
		self.alchemy_category_labeler_.fit(X.alchemy_category.tolist())
		return self		
	def transform(self, X):
		encoded_X = self.alchemy_category_labeler_.transform(X.alchemy_category.tolist())
		result_X = X.copy()
		for i,f in enumerate(self.alchemy_category_labeler_.classes_):
			if self.useful_features is None or f in self.useful_features:
				result_X[f] = encoded_X[:, i]
		return result_X[result_X.columns - ['alchemy_category']]
class NontextFeatures(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		"""X: pd.DataFrame
		"""
		nontext_features = [f for f in X.columns if f not in ['text', 'label']]
		return np.asarray(X[nontext_features])
class TargetLabel(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		"""
		X: pd.DataFrame
		"""
		return np.asarray(X['label'])
class TextFeatures(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		"""
		X: pd.DataFrame
		"""
		return np.asarray(X['text'])
class SoftThresholdFeatures(BaseEstimator, TransformerMixin):
	def __init__(self, n_features = 1500):
		self.n_features = n_features
		self.centers_ = None
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		"""X: list of texts
		"""
		n_samples = X.shape[0]
		#hasher = HashingVectorizer(encoding = 'utf-8', 
		#				stop_words='english', non_negative=True, 
		#				n_features=self.hash_size)
		#X_matrix = hasher.fit_transform(X)
		X_matrix = X
		if self.centers_ is None:
			selection = np.arange(n_samples)
			np.random.shuffle(selection)
			self.centers_ = X_matrix[selection[:self.n_features]]
		S = soft_threshold(X_matrix, self.centers_, thr = 'mean')
		return S
class Word2VecFeatures(BaseEstimator, TransformerMixin):
	def __init__(self, n_features, voc_file):
		self.n_features = n_features
		self.voc_file = voc_file
		self.word_clusters, self.grouped_words = self.read_word_cluster(voc_file)
		tfidf = TfidfVectorizer(encoding = 'iso-8859-1', stop_words='english')
		self.vectorize = tfidf.build_analyzer()
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		"""X: text data
		"""
		superword_counts = [Counter([self.word_clusters[w] 
									for w in self.vectorize(d) if w in self.word_clusters]) 
							for d in X]
		S = np.zeros((X.shape[0], self.n_features))
		for i, counter in enumerate(superword_counts):
			for icluster, counts in counter.iteritems():
				S[i, int(icluster)] = counts
		return S
	def read_word_cluster(self, fname):
		word_clusters = dict([line.strip().split() 
			for line in open(fname).readlines()])
		grouped_words = defaultdict(list)
		for w, c in word_clusters.iteritems():
			grouped_words[c].append(w)
		return word_clusters, grouped_words
class EnsembleModel(BaseEstimator, ClassifierMixin):
	def __init__(self, models):
		self.models = models
		self.classes_ = None
	def fit(self, X, y = None):
		for model in self.models:
			model.fit(X, y)
		self.classes_ = np.unique(y)
	def predict_proba(self, X):
		yhat = self.models[0].predict_proba(X)
		for i, model in enumerate(self.models[1], 1):
			yhat += model.predict_proba(X)
		return yhat / len(self.models)
	def predict(self, X):
		yhat = self.predict_proba(X)
		return self.classes_[np.argmax(yhat, axis = 1)]

def combine_predictions(yhats):
	yhat = yhats[0]
	for i, y in enumerate(yhats[1], 1):
		yhat += y
	return yhat / len(yhats)

def calculate_auc(y, yhat):
	fpr, tpr, thresholds = metrics.roc_curve(y, yhat[:, 1], pos_label=1)
	return metrics.auc(fpr, tpr)
		

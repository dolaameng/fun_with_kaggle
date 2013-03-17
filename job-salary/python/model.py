from sklearn.base import BaseEstimator
from joblib import Parallel, delayed
import numpy as np
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import neighbors
import sys
from feature_extractor import *
from sklearn.pipeline import Pipeline
from transformer import *
from sklearn import metrics


def make_BOW_feature_extractor(category):
    if category in ('Part time Jobs',):
        DescBOW = ('DescriptionBOW', BOWFeatureExtractor('FullDescription'))      
    else:
        DescBOW = ('DescriptionBOW', VocFeatureExtractor('FullDescription', ngram_range=(1,1), min_df = 0.5))
    if category in ('Maintenance Jobs', 'Part time Jobs'):
        LocBOW = ('Location', LocationFeatureExtractor())
    else:
        LocBOW = ('LocationBOW', BOWFeatureExtractor('LocationRaw')) 
    return SpaseFeatureUnion([
        #('Category', Pipeline(steps = [
        #	('CategoryOneHot', OneHotNAFeatureExtractor('Category')),
        #	('SGSelector', SGDFeatureSelector()),
        #])),
        #('Company', Pipeline(steps = [
        #	('CompanyOneHot', CompanyOneHotFeatureExtractor()),
        #	#('SGSelector', SGDFeatureSelector()),
        #])),
        ('ConractTime', Pipeline(steps =[ 
            ('ContractTimeImputator', ContractTimeImputator()),
            ('ContractTimeNA',OneHotNAFeatureExtractor('ContractTime')),
        ])),
        ('ContractType', Pipeline(steps = [
            ('ContractTypeImputator', ContractTypeImputator()),
            ('ContractTypeNA', OneHotNAFeatureExtractor('ContractType')),
        ])),
        ('Title', Pipeline(steps = [
            #('TitleBOW', BOWFeatureExtractor('Title', ngram_range = (1, 2), min_df = 2, max_df = 1.0)),
            ('TitleBOW', VocFeatureExtractor('Title', ngram_range=(1, 2), min_df=0.3, min_group_size=3)), 
            #('SGSelector', SGDFeatureSelector()),
        ])), 
        ('Desc', Pipeline(steps = [
            #('DescriptionSimplierf', DescriptionSimplifier()),
            #('DescriptionBOW', BOWFeatureExtractor('FullDescription')), 
            #('DescriptionBOW', VocFeatureExtractor('FullDescription', ngram_range=(1,1),min_df = 10, max_df = 0.95)),
            DescBOW,
            #('SGSelector', SGDFeatureSelector()),
        ])),
        #('LocationBOW', BOWFeatureExtractor('LocationRaw')),
        ('Location', LocationFeatureExtractor()),
        #LocBOW,
        #('LocationBOW', VocFeatureExtractor('LocationRaw', ngram_range=(1,1),min_df = 10, max_df = 0.95)) 
            #('TitleLevels', TitleClusterFeatureExtractor()),
    ])
	
def make_rf():
    return ensemble.RandomForestRegressor(n_estimators = 200, verbose = 0, n_jobs = -1, min_samples_split = 10, max_features = None)
    
def make_sgd():
    return linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001, l1_ratio=0.15, verbose=1)
    
def make_nn():
    return neighbors.KNeighborsRegressor(2, weights = 'uniform', p=1)
    
def make_rf_pipeline(category):
    return Pipeline(steps = [
	    ('features', make_BOW_feature_extractor(category)),
	    ('ToDense', ToDenseMatrix()),
	    ('rf', make_rf()),
	])
	
def make_sgd_pipeline():
    return Pipeline(steps = [
	    ('features', make_BOW_feature_extractor("")),
	    ('ToDense', ToDenseMatrix()),
	    ('sgd', make_sgd()),
	])
	
def make_nn_pipeline(category):
    bow = SpaseFeatureUnion([('FullDescription', BOWFeatureExtractor('FullDescription',max_features=100)),
                ('Title', BOWFeatureExtractor('Title',max_features=100)),
                ('LocationRaw', BOWFeatureExtractor('LocationRaw',max_features=100, ngram_range = (1, 3), min_df = 1, max_df = 1.0)),
                ('LocationNormalized', BOWFeatureExtractor('LocationNormalized',max_features=100, min_df = 1, max_df = 1.0)),
                ])
    return Pipeline(steps = [
	    #('features', make_BOW_feature_extractor(category)),
	    ('features', bow),
	    ('ToDense', ToDenseMatrix()),
	    ('nn', make_nn()),
	])


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
		
class CategorizedEnsemble(BaseEstimator):
	def __init__(self, field, ensemble_dict, fit_default = False):
	    """
	    ensemble_dict should include a "default" model
	    for fall-back matching
	    """
	    self.field = field
	    self.ensemble_dict = ensemble_dict
	    self.categories = self.ensemble_dict.keys()
	    self.default_model = self.ensemble_dict['default']
	    self.fit_default = fit_default
	def fit(self, X, y):
		## partition
		categorized_Xs = dict(list(X.groupby(self.field)))	
		categorized_ys = dict(map(lambda (name, data): (name, y[data.index]), categorized_Xs.items()))
		print "____________________________"
		print [XX.shape[0] for XX in categorized_Xs.values()]
		print "_____________________________"
		assert set(categorized_Xs.keys()) == set(categorized_ys.keys()) <= set(self.categories)
		#for (i,k) in enumerate([ 'Part time Jobs']):
		for (i,k) in enumerate(categorized_Xs.keys()):
		    print '------------training model for category ', i, k, '---------------', categorized_Xs[k].shape
		    self.ensemble_dict[k].fit(categorized_Xs[k], categorized_ys[k])	 
		if self.fit_default:
		    print '----------train default model------------'
		    self.default_model.fit(X, y) 
		return self
	def predict(self, X):
	    indices = X.index
	    categorized_Xs = dict(list(X.groupby(self.field)))
	    ys = []	
	    assert set(categorized_Xs.keys()) <= set(self.categories)
	    for (i,k) in enumerate(categorized_Xs.keys()):    
	        if k in self.ensemble_dict:
	            print '---------------predicting using model category',i, k, '--------------', categorized_Xs[k].shape
	            #print categorized_Xs[k].columns, categorized_Xs[k].shape
	            #print self.ensemble_dict[k]
	            #sys.exit(-1)
	            yhat = self.ensemble_dict[k].predict(categorized_Xs[k]) 
	            #print len(yhat), yhat
	            #print len(categorized_Xs[k].index)
	            yseries = pd.Series(yhat, index=categorized_Xs[k].index)
	            #print yseries
	            ys.append(yseries)
	        else:
	            print '---------using default model to predict-----------'
	            yhat = self.ensemble_dict[k].predict(categorized_Xs[k]) 
	            yseries = pd.Series(yhat, index=categorized_Xs[k].index)
	            ys.append(yseries)
	        print "ACCURACY:", k, ":", metrics.mean_absolute_error(categorized_Xs[k]['SalaryNormalized'], np.exp(yseries)) 
	    #print 'ys:',ys
	    #print 'indices:', indices
	    #print 'y:', pd.concat(ys)
	    
	    y = pd.concat(ys)[indices]
	    return y.tolist()
	def fit_predict(self, X, y):
	    return self.fit(X, y).predict(X)    
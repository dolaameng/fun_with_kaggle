from sklearn import cross_validation
from sklearn import ensemble
from sklearn import pipeline
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
from feature_extractor import *
from transformer import *
from model import *
import joblib

def test_single_feature(train, test, feature_extractor, target_transformer):
	(train_raw_X, train_raw_y) = (train, train['SalaryNormalized'])
	(test_raw_X, test_raw_y) = (test, test['SalaryNormalized'])


	print 'feature extraction ...'
	train_y = target_transformer.transform(train_raw_y)
	test_y = target_transformer.transform(test_raw_y)
	train_X = feature_extractor.fit_transform(train_raw_X, train_y)
	test_X = feature_extractor.transform(test_raw_X)

	print 'fit single feature ...'
	train_raw_yhat = target_transformer.r_transform(train_X)
	test_raw_yhat = target_transformer.r_transform(test_X)


	print 'evaluate error metrics ...'
	#train_error = metrics.mean_absolute_error(train_raw_y, train_raw_yhat)
	print test_raw_y.shape, test_raw_yhat.shape
	test_error = metrics.mean_absolute_error(test_raw_y, test_raw_yhat.reshape(-1,))
	#print 'Train error: ', train_error
	print 'Test error:', test_error

def test_regressor(train, test, feature_extractor, target_transformer, regressor):
	(train_raw_X, train_raw_y) = (train, train['SalaryNormalized'])
	(test_raw_X, test_raw_y) = (test, test['SalaryNormalized'])


	print 'feature extraction ...'
	train_y = target_transformer.transform(train_raw_y)
	test_y = target_transformer.transform(test_raw_y)
	train_X = feature_extractor.fit_transform(train_raw_X, train_y)
	test_X = feature_extractor.transform(test_raw_X)

	print 'fit regression model ...'
	try:
		regressor.fit(train_X, train_y)
		train_raw_yhat = target_transformer.r_transform(regressor.predict(train_X))
		test_raw_yhat = target_transformer.r_transform(regressor.predict(test_X))
	except TypeError:
		regressor.fit(train_X.toarray(), train_y)
		train_raw_yhat = target_transformer.r_transform(regressor.predict(train_X.toarray()))
		test_raw_yhat = target_transformer.r_transform(regressor.predict(test_X.toarray()))

	print 'evaluate error metrics ...'
	train_error = metrics.mean_absolute_error(train_raw_y, train_raw_yhat)
	test_error = metrics.mean_absolute_error(test_raw_y, test_raw_yhat)
	print 'Train error: ', train_error
	print 'Test error:', test_error

def test_model_only(train, test, regressor, target_transformer):
	(train_raw_X, train_raw_y) = (train, train['SalaryNormalized'])
	(test_raw_X, test_raw_y) = (test, test['SalaryNormalized'])

	train_y = target_transformer.transform(train_raw_y)
	test_y = target_transformer.transform(test_raw_y)
	#train_X = train_raw_X.copy()
	train_X = train_raw_X
	test_X = test_raw_X
	print "TRAIN SHAPE------------", train_X.shape

	print 'fit regression model ...'
	regressor.fit(train_X, train_y)
	print "TRAIN SHAPE------------", train_X.shape
	
	#train_X = train_raw_X.copy()
	#test_X = test_raw_X.copy()
	#print regressor.predict(test_X)
	#print regressor
	#train_raw_yhat = target_transformer.r_transform(regressor.predict(train_X))
	test_raw_yhat = target_transformer.r_transform(regressor.predict(test_X))
	print "TRAIN SHAPE------------", train_X.shape
    
    
	print 'evaluate error metrics ...'
	#train_error = metrics.mean_absolute_error(train_raw_y, train_raw_yhat)
	test_error = metrics.mean_absolute_error(test_raw_y, test_raw_yhat)
	#print 'Train error: ', train_error
	print 'Test error:', test_error

def main():
	## load raw data
	def identity(x): return x
	converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             , "Company": identity
             , 'ContractTime': identity
             , 'ContractType': identity
    }
	print 'load data ...'
	data = pd.read_csv('../data/Train_rev1.csv', header = 0, converters = converters).ix[:20000, :]
	valid = pd.read_csv('../data/Valid_rev1.csv', header = 0, converters = converters)
	ndata, nfeature = data.shape
	train_indices, test_indices = cross_validation.train_test_split(range(ndata), test_size = 0.2)
	print 'TEST SIZE:', len(test_indices)
	(train, test) = (data.ix[train_indices, :], data.ix[test_indices, :])
	## feature extraction of inputs
	"""
	bag_of_words = SpaseFeatureUnion(
					[
						('TitleBOW', BOWFeatureExtractor('Title')), 
						('DescriptionBOW', BOWFeatureExtractor('FullDescription')),
						#('LocationBOW', BOWFeatureExtractor('LocationNormalized')),
						('Location', LocationFeatureExtractor()),
						('ContractTimeNA',OneHotNAFeatureExtractor('ContractTime')),
						('ContractTypeNA', OneHotNAFeatureExtractor('ContractType')),
						('TitleLevels', TitleClusterFeatureExtractor())
					])
	"""
	bag_of_words = SpaseFeatureUnion(
					[
						#('Category', Pipeline(steps = [
						#	('CategoryOneHot', OneHotNAFeatureExtractor('Category')),
						#	('SGSelector', SGDFeatureSelector()),
						#])),
						#('Company', Pipeline(steps = [
						#	('CompanyOneHot', CompanyOneHotFeatureExtractor()),
						#	('SGSelector', SGDFeatureSelector()),
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
							('TitleBOW', BOWFeatureExtractor('Title', ngram_range = (1, 2), min_df = 2, max_df = 1.0)),
							#('TitleBOW', VocFeatureExtractor('Title')), 
							('SGSelector', SGDFeatureSelector()),
						])), 
						('Desc', Pipeline(steps = [
							#('DescriptionSimplierf', DescriptionSimplifier()),
							#('DescriptionBOW', BOWFeatureExtractor('FullDescription')), 
							('DescriptionBOW', VocFeatureExtractor('FullDescription', ngram_range=(1,1),min_df = 30, max_df = 0.95)),
							#('SGSelector', SGDFeatureSelector()),
						])),
						('LocationBOW', BOWFeatureExtractor('LocationRaw')),
						#('Location', LocationFeatureExtractor()),
						#('TitleLevels', TitleClusterFeatureExtractor()),
					])
	## transfomer for output - {identity, log_transformer}
	identity_transformer = IdentityTransformer()
	log_transformer = LogTransformer()
	## regressors
	rf = ensemble.RandomForestRegressor(n_estimators = 50, verbose = 0, n_jobs = -1, min_samples_split = 30)
	random_rf = RandomFeatureEnsemble(ensemble = [
			ensemble.RandomForestRegressor(n_estimators = 50, n_jobs = -1, min_samples_split = 30)
			for _ in xrange(5)
		], 
		n_features = 300)
	bt = ensemble.GradientBoostingRegressor(loss = 'lad', 
	    learning_rate= 0.1, n_estimators=200, subsample=0.5, max_depth=5, max_features=100,
	    verbose = 1)
	sgd = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.0001, l1_ratio=0.15, verbose=1)
	rf_pipeline = Pipeline(steps = [
	    ('features', make_BOW_feature_extractor("")),
	    ('ToDense', ToDenseMatrix()),
	    ('rf', make_rf()),
	])
	sgd_pipeline = Pipeline(steps = [
	    ('features', make_BOW_feature_extractor("")),
	    ('ToDense', ToDenseMatrix()),
	    ('sgd', make_sgd()),
	])
	
	categories = np.unique(data['Category']).tolist()
	categories.append('default')
	rf_category_ensemble = CategorizedEnsemble('Category', 
	    dict(zip(categories, [make_rf_pipeline(categories[i]) for i in xrange(len(categories)-1)] + [make_sgd_pipeline()])))
	nn_category_ensemble = CategorizedEnsemble('Category', 
    	    dict(zip(categories, [make_nn_pipeline(categories[i]) for i in xrange(len(categories)-1)] + [make_sgd_pipeline()])))
	#test_model_only(train, test, nn_category_ensemble, log_transformer)
	test_model_only(data, test, make_rf_pipeline(""), log_transformer)
	#print 'writing model...'
	#joblib.dump(rf_category_ensemble, '../models/salary.model', compress=6)
	## evaluate models
	print '-----------test random forest on bag of words-------'
	## LOG OF OUTPUT is better
	"""
	test_regressor(train, test, SpaseFeatureUnion([
		('title',InvertedIndexFeatureExtractor('Title')),
		('description',InvertedIndexFeatureExtractor('FullDescription')),]), identity_transformer, rf)
	"""
	#test_regressor(train, test, InvertedIndexFeatureExtractor('Title'), identity_transformer, rf)
	#test_single_feature(train, test, InvertedIndexFeatureExtractor('Title'), identity_transformer)
	#test_regressor(train, test, VocFeatureExtractor('Title'), identity_transformer, rf)
	#test_regressor(train, test, bag_of_words, log_transformer, rf)
	#test_regressor(train, test, bag_of_words, log_transformer, rf)
	"""
	test_regressor(train, test, Pipeline(steps = [
							('CompanyOneHot', CompanyOneHotFeatureExtractor()),
							('SGSelector', SGDFeatureSelector()),
						]), log_transformer, rf)
	"""
	#test_regressor(train, test, bag_of_words, log_transformer, random_rf)



if __name__ == '__main__':
	main()

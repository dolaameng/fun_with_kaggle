from sklearn import cross_validation
from sklearn import ensemble
from sklearn import pipeline
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pandas as pd
from feature_extractor import *
from transformer import *
from model import *

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



def main():
	## load raw data
	def identity(x): return x
	converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             , "Company": identity
    }
	print 'load data ...'
	data = pd.read_csv('../data/Train_rev1.csv', header = 0, converters = converters)
	valid = pd.read_csv('../data/Valid_rev1.csv', header = 0, converters = converters)
	ndata, nfeature = data.shape
	train_indices, test_indices = cross_validation.train_test_split(range(ndata), test_size = 0.3)
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
						('Category', Pipeline(steps = [
							('CategoryOneHot', OneHotNAFeatureExtractor('Category')),
							('SGSelector', SGDFeatureSelector()),
						])),
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
							('SGSelector', SGDFeatureSelector()),
						])), 
						('Desc', Pipeline(steps = [
							('DescriptionSimplierf', DescriptionSimplifier()),
							('DescriptionBOW', BOWFeatureExtractor('FullDescription')), 
							('SGSelector', SGDFeatureSelector()),
						])),
						#('LocationBOW', BOWFeatureExtractor('LocationNormalized')),
						('Location', LocationFeatureExtractor()),
						#('TitleLevels', TitleClusterFeatureExtractor())
					])
	## transfomer for output - {identity, log_transformer}
	identity_transformer = IdentityTransformer()
	log_transformer = LogTransformer()
	## regressors
	rf = ensemble.RandomForestRegressor(n_estimators = 50, verbose = 2, n_jobs = -1, min_samples_split = 30)
	random_rf = RandomFeatureEnsemble(ensemble = [
			ensemble.RandomForestRegressor(n_estimators = 50, n_jobs = -1, min_samples_split = 30)
			for _ in xrange(5)
		], 
		n_features = 300)
	## evaluate models
	print '-----------test random forest on bag of words-------'
	## LOG OF OUTPUT is better
	#test_regressor(train, test, bag_of_words, identity_transformer, rf)
	test_regressor(train, test, bag_of_words, log_transformer, rf)
	"""
	test_regressor(train, test, Pipeline(steps = [
							('CompanyOneHot', CompanyOneHotFeatureExtractor()),
							('SGSelector', SGDFeatureSelector()),
						]), log_transformer, rf)
	"""
	#test_regressor(train, test, bag_of_words, log_transformer, random_rf)



if __name__ == '__main__':
	main()

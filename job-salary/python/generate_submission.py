## generate first if it does not exist
## evaluate model on valid file

import joblib 
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

MODEL_FILE = '../models/salary.model'
TARGET_TRANSFORMER = LogTransformer() # or IdentityTransformer()

def build_model():
    def identity(x): return x
    converters = { "FullDescription" : identity
             , "Title": identity
             , "LocationRaw": identity
             , "LocationNormalized": identity
             , "Company": identity
             , 'ContractTime': identity
             , 'ContractType': identity
    }
    print 'load train data ...'
    data = pd.read_csv('../data/Train_rev1.csv', header = 0, converters = converters)#.ix[:10000, :]
    print 'build model'
    categories = np.unique(data['Category']).tolist()
    categories.append('default')
    regressor = CategorizedEnsemble('Category', 
        dict(zip(categories, [make_rf_pipeline() for _ in xrange(len(categories)-1)] + [make_sgd_pipeline()])))
    (train_raw_X, train_raw_y) = (data, data['SalaryNormalized'])
    train_y = TARGET_TRANSFORMER.transform(train_raw_y)
    train_X = train_raw_X
    regressor.fit(train_X, train_y)
    #train_raw_yhat = TARGET_TRANSFORMER.r_transform(regressor.predict(train_X))
    #print 'evaluate error metrics ...'
    #train_error = metrics.mean_absolute_error(train_raw_y, train_raw_yhat)
    
    return regressor

def main():
    ## load or train the model
    try:
        model = joblib.load(MODEL_FILE)
    except IOError:
        model = build_model()
        print 'dump model...'
        joblib.dump(model, MODEL_FILE, compress = 3)
        

if __name__ == '__main__':
    main()
## build different classifiers and their ensembles

import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn import ensemble

from build_design_matrix import *

TRAIN_FILE = '../data/Train_rev1.csv'
VALID_FILE = '../data/Valid_rev1.csv'

def build_SGDRegressor(train_X, train_y, test_X, test_y):
    ##########
    log_train_y = np.log(train_y)
    ##########
    sgd_regressor = linear_model.SGDRegressor(loss='huber', penalty='l1', alpha=0.001, l1_ratio=0.15, verbose=True, n_iter = 50)
    sgd_regressor.fit(train_X, log_train_y)
    train_yhat = np.exp(sgd_regressor.predict(train_X))
    test_yhat = np.exp(sgd_regressor.predict(test_X))
    print metrics.mean_absolute_error(train_y, train_yhat)
    print metrics.mean_absolute_error(test_y, test_yhat)
    ## write to pickle
    
def build_boostedTree(train_X, train_y, test_X, test_y):
    bt = ensemble.GradientBoostingRegressor(loss = 'lad', 
        learning_rate= 0.1, n_estimators=100, subsample=0.3, max_depth=3, max_features=50, 
        verbose = 1)
    bt_train_X = train_X
    bt_test_X = test_X
    bt.fit(bt_train_X.toarray(), train_y)
    train_yhat = sgd_regressor.predict(bt_train_X)
    test_yhat = sgd_regressor.predict(bt_test_X)
    print metrics.mean_absolute_error(train_y, train_yhat)
    print metrics.mean_absolute_error(test_y, test_yhat)

def main():
    data = pd.read_csv(TRAIN_FILE, header = 0)
    X, _ = build_X(data)
    y = np.array(data.SalaryNormalized.tolist())
    train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y, test_size= 0.3)
    print train_X.shape, test_X.shape
    print train_y.shape, test_y.shape
    build_SGDRegressor(train_X, train_y, test_X, test_y)
    #build_boostedTree(train_X, train_y, test_X, test_y)

if __name__ == "__main__":
    main()
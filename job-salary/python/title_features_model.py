## Generate the model for feature extraction of title data
import sys, os.path, pickle
import numpy as np
import pandas as pd

from sklearn.feature_extraction import text
from sklearn import metrics
from sklearn import cluster
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import kernel_approximation
from sklearn.metrics import pairwise
from sklearn import ensemble
from sklearn import linear_model
from scipy import sparse
from sklearn import cross_validation
from sklearn import svm
from sklearn.base import BaseEstimator

TITLE_FILE = '../data/titles.csv'
#SALARY_FILE = '../data/salaries.csv'
#TRAIN_SIZE = 244768

class TitleFeatureExtractor(BaseEstimator):
    def __init__(self, n_clusters = 50, pca_n_components = 20, 
                        kmpca_n_components = 3, kernel_n_components = 30):
        self.counter = text.CountVectorizer(stop_words = 'english', 
                        ngram_range = (1, 2), 
                        min_df = 30, binary = True)
        self.km = cluster.MiniBatchKMeans(n_clusters = n_clusters, 
                    n_init=10, batch_size=10000, verbose = 1)
        self.pca = decomposition.RandomizedPCA(n_components=pca_n_components)
        self.kmpca = decomposition.RandomizedPCA(n_components=kmpca_n_components)
        self.rbf = kernel_approximation.RBFSampler(n_components=kernel_n_components)
        self.tree_hasher = ensemble.RandomTreesEmbedding(n_estimators=30, max_depth=5, n_jobs=4)
        self.X_names = ['CounterX', 'ClusterdX', 'KmX', 'PCAX', 'PCAClusterdX', 'RbfX', 'TreeX']
    ## X is the corpus - list of sentences
    def fit(self, X, y=None):
        self.counter.fit(X)
        CounterX = self.counter.transform(X)
        self.km.fit(CounterX)
        labels = self.km.labels_.reshape(-1, 1)
        ClusterdX = preprocessing.OneHotEncoder().fit_transform(labels)
        self.pca.fit(CounterX)
        KmX = self.km.transform(CounterX)
        self.kmpca.fit(KmX)
        self.rbf.fit(CounterX)
        self.tree_hasher.fit(ClusterdX.todense())
        return self
    def transform(self, X):
        # transform
        CounterX = self.counter.transform(X)
        labels = self.km.labels_.reshape(-1, 1)
        ClusterdX = preprocessing.OneHotEncoder().fit_transform(labels)
        KmX = self.km.transform(CounterX)
        PCAX = self.pca.transform(CounterX)
        PCAClusterdX = self.kmpca.transform(KmX)
        RbfX = self.rbf.transform(CounterX)
        TreeX = self.tree_hasher.transform(ClusterdX.todense())
        # index of transformed matrix
        Xs = [CounterX, ClusterdX, KmX, PCAX, PCAClusterdX, RbfX, TreeX]
        #feature_indices = np.array([M.shape[1] for M in Xs])
        #fs = np.cumsum(feature_indices).tolist()
        #self.feature_ranges = zip(X_names, zip([0]+fs, fs))
        X = sparse.hstack(Xs)
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def main():
    if len(sys.argv) < 2:
        print 'Usage: %s title_features.pickle' % (sys.argv[0],)
        sys.exit(-1)
    MODEL_FILE = sys.argv[1]
    if os.path.exists(MODEL_FILE):
        print 'model file already exists'
        sys.exit(-1)
    ## write out model
    title_feature_extractor = TitleFeatureExtractor()
    titles = np.array(pd.read_csv('../data/titles.csv', header = 0).Title.tolist())
    title_feature_extractor.fit(titles)
    pickle.dump(title_feature_extractor, open(MODEL_FILE, 'w'))
    print 'title_feature model successfully generated ...'
    

if __name__ == '__main__':
    main()
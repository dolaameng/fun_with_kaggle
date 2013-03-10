## Generate the model for feature extraction of location data
## NOTE the corpus is based on both train and validate files
import pandas as pd
import numpy as np
import csv, re, sys, os.path, pickle

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

from correlation_feature_selection import CorrelationFeatureSelector

RAW_AND_NORMALIZED_LOCATIONS_FILE = '../data/raw_and_normalized_locations.csv'
LOCATION_TREE_FILE = '../data/Location_Tree.csv'
TRAIN_FILE = '../data/Train_rev1.csv'

class LocationFeatureExtractor(BaseEstimator):
    def __init__(self, n_clusters = 100, pca_n_components = 10, 
                        kmpca_n_components = 7, kernel_n_components = 30):
        self.counter = text.CountVectorizer(stop_words = 'english', 
                                ngram_range = (1, 1), min_df = 2, max_df = 0.8, 
                                binary = True, lowercase=True)
        self.km = cluster.MiniBatchKMeans(n_clusters = n_clusters, 
                    n_init=10, batch_size=10000, verbose = 1)
        self.pca = decomposition.RandomizedPCA(n_components=pca_n_components)
        self.kmpca = decomposition.RandomizedPCA(n_components=kmpca_n_components)
        self.rbf = kernel_approximation.RBFSampler(n_components=kernel_n_components)
        self.tree_hasher = ensemble.RandomTreesEmbedding(n_estimators=30, max_depth=5, n_jobs=4)
        self.X_names = ['Loc_CounterX', 'Loc_ClusterdX', 'Loc_KmX', 'Loc_PCAX', 'Loc_PCAClusterdX', 'Loc_RbfX', 'Loc_TreeX']
        self.linear_feature_selector = None
        ## BUILD dictionary based on location_tree - faster for search
        location_tree = [row[0].lower().split('~')[::-1] for row in csv.reader(open(LOCATION_TREE_FILE))]
        self.location_dict = {}
        for locs in location_tree:
            for i in range(len(locs)):
                if locs[i] not in self.location_dict:
                    self.location_dict[locs[i]] = locs[i:]
    ## X is the corpus - list of sentences
    def fit(self, X, y=None):
        ## preprocessing
        (raw_locations, locations) = X
        ## BUILD first term of raw_location -- more informative than normalized location
        pattern = re.compile(r'\b[\w\s]+')
        locations_initial = map(lambda s: pattern.findall(s), raw_locations)
        locations_initial = map(lambda (i,s): s[0] if (s and s[0] in self.location_dict) else locations[i],
                                    enumerate(locations_initial))
        ## strategy: pick the first word from LocationRaw, 
        ## if found in the tree, use it, otherwise use LocationNormlalized
        ## the above data structure could be slow when searching
        standard_locations = np.array([' '.join(self.location_dict[initial]) if initial in self.location_dict 
                                                                        else 'UK' 
                                        for initial in locations_initial]) ## CAPTITAL 'UK' for unfound
        ## build feature extractors
        X = standard_locations
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
        if y is not None:
            self.linear_feature_selector = CorrelationFeatureSelector(pvalue_threshold=0.8)
            self.linear_feature_selector.fit(CounterX.tocsr()[:y.shape[0], :], y)
        return self
    def transform(self, X):
        ## preprocessing
        (raw_locations, locations) = X
        ## BUILD first term of raw_location -- more informative than normalized location
        pattern = re.compile(r'\b[\w\s]+')
        locations_initial = map(lambda s: pattern.findall(s), raw_locations)
        locations_initial = map(lambda (i,s): s[0] if (s and s[0] in self.location_dict) else locations[i],
                                    enumerate(locations_initial))
        ## strategy: pick the first word from LocationRaw, 
        ## if found in the tree, use it, otherwise use LocationNormlalized
        ## the above data structure could be slow when searching
        standard_locations = np.array([' '.join(self.location_dict[initial]) if initial in self.location_dict 
                                                                        else 'UK' 
                                        for initial in locations_initial]) ## CAPTITAL 'UK' for unfound
        ## build feature extractors
        X = standard_locations
        # transform
        CounterX = self.counter.transform(X)
        #labels = self.km.labels_.reshape(-1, 1)
        labels = self.km.predict(CounterX).reshape(-1, 1)
        ClusterdX = preprocessing.OneHotEncoder().fit_transform(labels)
        KmX = self.km.transform(CounterX)
        PCAX = self.pca.transform(CounterX)
        PCAClusterdX = self.kmpca.transform(KmX)
        RbfX = self.rbf.transform(CounterX)
        TreeX = self.tree_hasher.transform(ClusterdX.todense())
        # index of transformed matrix
        if self.linear_feature_selector:
            CounterX = self.linear_feature_selector.transform(CounterX)
        Xs = [CounterX, ClusterdX, KmX, PCAX, PCAClusterdX, RbfX, TreeX]
        fs = [M.shape[1] for M in Xs]
        self.feature_names_ = sum([map(lambda fi: self.X_names[i]+'_'+str(fi), range(findex)) 
                                    for (i,findex) in enumerate(fs)], 
                                    [])
        ## result matrix
        X = sparse.hstack(Xs)
        return X
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def main():
    if len(sys.argv) < 2:
        print 'Usage: %s location_features.pickle' % (sys.argv[0],)
        sys.exit(-1)
    MODEL_FILE = sys.argv[1]
    if os.path.exists(MODEL_FILE):
        print 'model file already exists'
        sys.exit(-1)
    
    df = pd.read_csv(RAW_AND_NORMALIZED_LOCATIONS_FILE, header = 0)
    raw_locations = np.array(map(lambda s: s.lower(), df.LocationRaw.tolist()))
    locations = np.array(map(lambda s: s.lower(), df.LocationNormalized.tolist()))
    

    ## write out model
    location_feature_extractor = LocationFeatureExtractor()
    y = np.log(np.array(pd.read_csv(TRAIN_FILE, header = 0).SalaryNormalized.tolist()))
    location_feature_extractor.fit([raw_locations, locations], y)
    pickle.dump(location_feature_extractor, open(MODEL_FILE, 'w'))
    print 'location_feature model successfully generated ...'
    

if __name__ == '__main__':
    main()
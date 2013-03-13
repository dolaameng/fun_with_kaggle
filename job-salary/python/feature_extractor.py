## build feature_extractor model
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import text
from sklearn import feature_extraction
from sklearn import cluster
from joblib import Parallel, delayed
from scipy import sparse
import re, csv, sys
import numpy as np


class BOWFeatureExtractor(BaseEstimator):
    def __init__(self, field, max_features = 100, selected_features = None):
        ## max_features selects features based on their term frequences
        ## it makes a HUGE difference from randomly selecting the features
        self.counter = text.CountVectorizer(stop_words = 'english', 
                        ngram_range = (1, 1), binary = True, 
                        max_features = max_features,
                        lowercase = True)
        self.field = field
        self.selected_features = selected_features
    def fit(self, X, y=None):
        self.counter.fit(X[self.field])
        return self
    def transform(self, X):
        result = self.counter.transform(X[self.field])
        if self.selected_features:
            result = result.tocsc()[:, self.selected_features]
        return result
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

        
class LocationFeatureExtractor(BaseEstimator):
    def __init__(self, max_features = 100):
        self.field = 'ProcessedLocation'
        self.counter = BOWFeatureExtractor(self.field, max_features = max_features)
        self.LOCATION_TREE_FILE = '../data/Location_Tree.csv'
        ## BUILD dictionary based on location_tree - faster for search
        location_tree = [row[0].lower().split('~')[::-1] for row in csv.reader(open(self.LOCATION_TREE_FILE))]
        self.location_dict = {}
        for locs in location_tree:
            for i in range(len(locs)):
                if locs[i] not in self.location_dict:
                    self.location_dict[locs[i]] = locs[i:]
    def fit(self, X, y=None):
        X = self._preprocess(X)
        self.counter.fit(X, y)
        return self
    def transform(self, X):
        X = self._preprocess(X)
        return self.counter.transform(X)
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    def _preprocess(self, X):
        ## preprocessing
        (raw_locations, locations) = X['LocationRaw'].tolist(), X['LocationNormalized'].tolist()
        ## BUILD first term of raw_location -- more informative than normalized location
        pattern = re.compile(r'\b[\w\s]+')
        locations_initial = map(lambda s: pattern.findall(s), raw_locations)
        locations_initial = map(lambda (i,s): s[0] if (s and s[0] in self.location_dict) else locations[i],
                                        enumerate(locations_initial))
        ## strategy: pick the first word from LocationRaw, 
        ## if found in the tree, use it, otherwise use LocationNormlalized
        ## the above data structure could be slow when searching
        processed_locations = np.array([' '.join(self.location_dict[initial]) 
                                                if initial in self.location_dict 
                                                else 'UK' 
                                            for initial in locations_initial]) ## CAPTITAL 'UK' for unfound
        X[self.field] = processed_locations
        return X

class OneHotNAFeatureExtractor(BaseEstimator):
    def __init__(self, field, na_string = "nan", na_class = 'miss'):
        self.na_string = na_string
        self.na_class = na_class
        self.field = field
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        sub_X = np.array(X[self.field].tolist())
        sub_X[sub_X==self.na_string] = self.na_class
        sub_X = [{self.field:v} for v in sub_X]
        dicter = feature_extraction.DictVectorizer()
        sub_X = dicter.fit_transform(sub_X)
        return sub_X
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
        

def TitleClusterFeatureExtractor_trans(level, salary, titles, counter): 
    return (counter.transform(titles), salary)   
def TitleClusterFeatureExtractor_clus(CounterX, salary): 
    return (cluster.MiniBatchKMeans(n_clusters = 1).fit(CounterX), salary) 
def TitleClusterFeatureExtractor_dist(km, CounterX): 
    return km.transform(CounterX)   
class TitleClusterFeatureExtractor(BaseEstimator):
    def __init__(self, n_levels = 1000):
        self.n_levels = n_levels
        self.counter = text.CountVectorizer(stop_words = 'english', binary = True, charset_error='ignore',
                                lowercase = True, min_df=1, charset='utf-8', token_pattern=r'\b[a-z][a-z]+\b', 
                                #max_features = 500,
                        )
    def fit(self, X, y=None):
        """y could be transformed or not
        """
        self.counter.fit(X['Title'])
        min_y, max_y = y.min(), y.max()
        X['SalaryLevels'] = np.digitize(y.tolist(), 
                    bins = np.linspace(min_y, max_y, (max_y-min_y)/self.n_levels+1))
        X['TitleY'] = y
        salary_groups = X.groupby('SalaryLevels')
        title_groups = [(salary_level, 
                        groups['TitleY'].mean(), 
                        ' '.join(map(str, groups['Title'].tolist()))) 
                            for (salary_level, groups) in salary_groups]
        """
        CounterX_groups = Parallel(n_jobs=-1)(delayed(TitleClusterFeatureExtractor_trans)
                                                          (level, salary, titles, self.counter) 
                                        for (level, salary, titles) in title_groups)
        self.ClusterX_groups = Parallel(n_jobs=-1)(delayed(TitleClusterFeatureExtractor_clus)(CounterX, salary) 
                        for (CounterX, salary) in CounterX_groups)
        """
        CounterX_groups = [TitleClusterFeatureExtractor_trans(level, salary, titles, self.counter) 
                                                        for (level, salary, titles) in title_groups]
        self.ClusterX_groups = [TitleClusterFeatureExtractor_clus(CounterX, salary) 
                                                        for (CounterX, salary) in CounterX_groups]
        
        self.clusters = np.array(map(lambda (km, salary): km, self.ClusterX_groups))
        self.salaries = np.array(map(lambda (km, salary): salary, self.ClusterX_groups))
        
        return self
    def transform(self, X):
        CounterX = self.counter.transform(X['Title'])
        distances = Parallel(n_jobs=-1)(delayed(TitleClusterFeatureExtractor_dist)(km, CounterX) 
                        for km in self.clusters)
        #print distances_salary[0][0]
        #print distances_salary[0][0].shape
        dist_mat = np.hstack(distances)
        cluster_labels = np.argmin(dist_mat, axis = 1)
        return self.salaries[cluster_labels].reshape(-1, 1)
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def feature_fit(name, transformer, X, y = None):
    return (name, transformer.fit(X, y))
def feature_transform(transformer, X):
    return transformer.transform(X)
class SpaseFeatureUnion(BaseEstimator):
    def __init__(self, transformers):
        self.transformers = transformers
    def fit(self, X, y = None):
        self.transformers = Parallel(n_jobs = -1)(delayed(feature_fit)(name, transformer, X, y) 
                                                for (name, transformer) in self.transformers)
        return self
    def transform(self, X):    
        results = Parallel(n_jobs = -1)(delayed(feature_transform)(transformer, X) 
                                        for (name, transformer) in self.transformers)
        merged = sparse.hstack(results)
        return merged
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
        
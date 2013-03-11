## generate design matrix for train or test data
## using different feature extracted built from other scripts
## e.g. title_features.pickle, description_features.pickle and location_features.pickle

## this script will create a lot of features - use feature selection script to select
## significant features and then train models on it

import sys, os.path, pickle
from scipy import io
import pandas as pd
from scipy import sparse
import numpy as np

from description_features_model import DescriptionFeatureExtractor
from title_features_model import TitleFeatureExtractor
from location_features_model import LocationFeatureExtractor
from contracttype_features_model import ContractTypeFeatureExtractor
from contracttime_features_model import ContractTimeFeatureExtractor

TITLE_MODEL = '../models/title_features.pickle'
DESC_MODEL = '../models/description_features.pickle'
LOC_MODEL = '../models/location_features.pickle'

TRAIN_FILE = '../data/Train_rev1.csv'
VALID_FILE = '../data/Valid_rev1.csv'

def main():
    if len(sys.argv) < 3:
        print 'Usage: python %s {train|test} matrix_file.pickle' % (sys.argv[0],)
        sys.exit(-1)
    MODE, MATRIX_FILE = sys.argv[1:]
    if os.path.exists(MATRIX_FILE):
        print 'design matrix file already exists - remove it first'
        sys.exit(-1)
    
    ## load data
    print 'loading data ...'
    if MODE == 'train':
        data = pd.read_csv(TRAIN_FILE, header = 0)
    else:
        data = pd.read_csv(VALID_FILE, header = 0)
        
    (FEATS, FEAT_NAMES) = build_X(data)
    ## write out
    print 'write matrix to mtx format'
    #io.mmwrite(MATRIX_FILE, FEATS)
    #pickle.dump((FEATS, FEAT_NAMES), open(MATRIX_FILE, 'w'), protocol=2)
    #pickle.dump(FEATS, open(MATRIX_FILE, 'w'), protocol=2)
    

def build_X(data):   
    ## title related features
    print 'generating title features ...'
    title_model = pickle.load(open(TITLE_MODEL))
    TITLES = title_model.transform(np.array(data.Title.tolist()))
    TITLE_NAMES = title_model.feature_names_
    
    ## description related features
    print 'generating description features ...'
    desc_model = pickle.load(open(DESC_MODEL))
    DESCRIPTIONS = desc_model.transform(data.FullDescription.tolist())
    DESCRIPTION_NAMES = desc_model.feature_names_
    
    ## location related features
    print 'generating location features ...'
    loc_model = pickle.load(open(LOC_MODEL))
    raw_locations = np.array(map(lambda s: s.lower(), data.LocationRaw.tolist()))
    locations = np.array(map(lambda s: s.lower(), data.LocationNormalized.tolist()))
    LOCATIONS = loc_model.transform([raw_locations, locations])
    LOCATION_NAMES = loc_model.feature_names_
    
    ## contract type related features
    print 'generating contract type features ...'
    contype_model = ContractTypeFeatureExtractor()
    CONTRACT_TYPES = contype_model.fit_transform(data.ContractType.tolist())
    CONTRACT_TYPES_NAMES = contype_model.feature_names_
    
    ## contract time related features
    print 'generating contract time features ...'
    contime_model = ContractTimeFeatureExtractor()
    CONTRACT_TIMES = contime_model.fit_transform(data.ContractTime.tolist())
    CONTRACT_TIMES_NAMES = contime_model.feature_names_
    
    ## company related features
    ## TODO
    
    ## category related features
    ## TODO
    
    ## sourcename related features (with ??)
    ## TODO
    
    ## merge design matrix (and optionally target)
    print 'merging data ...'
    FEATS = sparse.hstack([TITLES, DESCRIPTIONS, LOCATIONS, CONTRACT_TYPES, CONTRACT_TIMES])
    print 'generate ', type(FEATS), 'type matrix ...'
    print 'matrix shape', FEATS.shape
    FEAT_NAMES = TITLE_NAMES + DESCRIPTION_NAMES + LOCATION_NAMES + CONTRACT_TYPES_NAMES + CONTRACT_TIMES_NAMES
    
    return (FEATS, FEAT_NAMES)
    


if __name__ == "__main__":
    main()
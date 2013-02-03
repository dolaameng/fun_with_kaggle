## convert the train.csv to user_events_pref.csv

SRC_PATH = "../data/train.csv"
DES_PATH = "../data/train_user_events_pref.csv"

import pandas as pd
import numpy as np

## load data
data = pd.read_csv(SRC_PATH, header = 0, 
                    na_values = ["", "None"], parse_dates = ["timestamp"])
"""                    
## calculate preferences based on "interested", "not_interested" (and "invited"?)
## (not_interested, interested) = 
## (1, 0) 0
## (0, 0) 1
## (0, 1) 12
## (1, 1) impossible
data['preference'] = data.apply(lambda r: 0 
                            if r['not_interested']==1 
                            else (2 if r['interested']==1 else 1), 
                    axis=1)
"""
## calculate preferences based on "interested", "not_interested" (and "invited"?)
## (not_interested, interested) = 
## (1, 0) 0
## (0, 0) absent (-1 and them remove)
## (0, 1) 1
## (1, 1) impossible
data['preference'] = data.apply(lambda r: 0 
                            if r['not_interested']==1 
                            else (1 if r['interested']==1 else np.nan), 
                    axis=1)
data = data[pd.notnull(data['preference'])]
                    
## write to preference file
data.to_csv(DES_PATH, na_rep="", 
            header = False, index = False, 
            cols = ["user", "event", "preference"])
            
print 'done with converting to ', DES_PATH
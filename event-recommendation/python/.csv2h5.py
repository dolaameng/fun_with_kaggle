## so far convert events.csv to h5 only
import pandas as pd
import sys

NROWS = 3137972
STEP = 500000

events = pd.read_csv('../data/events.csv', header = 0,  
                    na_values=["NA", "None", ""],  
                    parse_dates=['start_time'], iterator=True)


events_hf5 = pd.HDFStore('../data/events.h5', mode='w', complevel=9, )
total = 0     
               
while True:
    print '%d rows processed: %f%%' % (total, total*100./NROWS)
    try:
        chunk = events.get_chunk(STEP)
        events_hf5.append('events', chunk, 
                        min_itemsize = { 'values' : 74 }, 
                        nan_rep='nan', data_columns=True)
    except RuntimeError as ex:
        print ex
        break
    total += STEP
    
events_hf5.close()



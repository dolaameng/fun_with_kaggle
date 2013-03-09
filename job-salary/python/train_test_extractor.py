## collect train and test certain columns (e.g. titles) into one file 
## in the order that they appear in the train and test files

import pandas as pd
import json
import sys

def main():
    if len(sys.argv) < 3:
        print 'Usage: python %s dest_file field1 field2 ...' % (sys.argv[1], )
        sys.exit(-1)
    dest_file = sys.argv[1]
    fields = sys.argv[2:]
    setting = json.load(open('settings.json'))
    train_file, test_file = setting['raw_train_path'], setting['raw_valid_path']
    train_df = pd.read_csv(train_file, header = 0)
    test_df = pd.read_csv(test_file, header = 0)
    merged_df = pd.concat([train_df.ix[:, fields], test_df.ix[:, fields]], axis = 0)
    merged_df.to_csv(dest_file, header = True, index = False)

if __name__ == '__main__':
    main()
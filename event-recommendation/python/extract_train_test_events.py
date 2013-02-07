## extract events information contained in train.csv and test.csv from events.csv

import sys
import pandas as pd
import csv

def main():
    train_path = '../data/train.csv'
    test_path = '../data/test.csv'
    events_path = '../data/events.csv'
    train_test_events_path = '../data/train_test_events.csv'
    
    train_event_ids = set(pd.read_csv(train_path, header=0)['event'].tolist())
    test_event_ids = set(pd.read_csv(test_path, header=0)['event'].tolist())
    event_ids = train_event_ids.union(test_event_ids)
    
    with open(events_path, 'r') as fevents:
        reader = csv.reader(fevents)
        data_headers = reader.next()
        data_body = [
            line
            for line in reader
            if int(line[0]) in event_ids
        ]
    data = pd.DataFrame(data_body, columns = data_headers)
    data.to_csv(train_test_events_path, header = True, index = False)
    print 'generated file', train_test_events_path

if __name__ == '__main__':
    main()
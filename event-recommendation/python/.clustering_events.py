import os, pickle
import numpy as np

def generate_event_topic_file():
    print 'cutting events.csv to extract common words frequence'
    ## headers include
    ## (event_id,c_1,...,c_100,c_other)
    os.system('cat ../data/events.csv | cut -d , -f 10- > ../data/events_words.csv')
    os.system('cat ../data/events.csv | cut -d , -f 1 > ../data/events_ids.csv')
    #os.system('tail -n+2 events.csv | cut -d , -f 1,10- > events_topic.csv')
    #print 'write event topic as python pickle data'

def main():
    generate_event_topic_file()
    
    
if __name__ == '__main__':
    main()
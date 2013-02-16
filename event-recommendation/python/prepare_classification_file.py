## generate train/test file for classification model by
## extracting messages from train.csv, expanding user_id with
## information from users.csv/user_friends.csv, and expanding event_id with 
## information from events.csv

## details:
## (1) In the train.csv file, user will be expanded as
## (user_id, locale, birthyear, gender, joinedAt, location, timezone, friend_with_creator)
## (2) In the train.csv file, event will be expanded as
## (event_id, start_time, city, state, zip, country, lat, lng, c_1, ... c_100, c_other)
## (3) In the train.csv file, the following fields will remain
## invited, timestamp, interested, not_interested (the last two are
## potential targets)
## some fields may not necessarily be related to classification, 
## because there are a lot of missing values in it.
## but better keep them in the data set first.

## read original_path and dest_path from command line

#original_path = None #'../data/train.csv'
#dest_path = None #'../data/train_classifier.csv'
users_path = '../data/users.csv'
events_path = '../data/events.csv' # large file
friends_path = '../data/user_friends.csv'
attendence_path = '../data/event_attendees.csv'
clusters_path = '../data/event_clusters32.csv'
communities_path = '../data/user_communities.csv'
na_values = ['None', ' ', 'NA', '']
normalized_na = 'NA'


import numpy as np
import pandas as pd
from pandas import DataFrame
import csv, sys
from dateutil.parser import parse

def load_train_csv(original_path):
    # no need for index / data parsing
    ## headers for train data:
    ## (user,event,invited,timestamp,interested,not_interested)
    train = pd.read_csv(original_path, header=0, na_values=na_values)
    return train
    
def fill_train_with_events(train):
    ## fill event field in train with 
    ## (event, start_time, city, state, zip, country, lat, lng, c_1, ... c_100, c_other)
    ## in the events.csv file
    ## headers for events.csv file - event_id in each row is UNIQUE
    ## (event_id,user_id,start_time,city,state,zip,country,lat,lng,
    ## c_1, ... c_100, c_other)
    ## build fast event_id indexing from train
    event_set = set(train['event']) ## 8846 events in train
    with open(events_path, 'r') as fevents:
        reader = csv.reader(fevents)
        event_headers = reader.next() # header
        event_headers = ['event', 'event_creator'] + ["event_"+h for h in event_headers[2:]]
        event_data = [[int(row[0])]+row[1:] for row in reader 
                    if int(row[0]) in event_set]
    events = DataFrame(event_data, columns = event_headers)
    return pd.merge(train, events)
    
def fill_train_with_users(train):
    ## fill user filed in train with
    ## (user_id,locale,birthyear,gender,joinedAt,location,timezone)
    ## like in events.csv, the user_id in users.csv is unique
    ## build user_id indexing from train
    user_set = set(train['user']) ## 2034 users in train
    with open(users_path, 'r') as fusers:
        reader = csv.reader(fusers)
        user_headers = reader.next() 
        user_headers = ['user'] + ["user_"+h for h in user_headers[1:]]
        user_data = [[int(row[0])]+row[1:] for row in reader
                    if int(row[0]) in user_set]
    users = DataFrame(user_data, columns = user_headers)
    return pd.merge(train, users)
    
def fill_train_with_communities(train):
    user_communities = pd.read_csv(communities_path, header = 0)
    return pd.merge(train, user_communities, how='left')
    
def fill_train_with_attendence(train):
    ## extract event attendence information (popularity of event)
    ## (or with user information ?? -- very sparse overlapping)
    ## in event_attendees.csv, headers are (24144 events)
    ## event,yes,maybe,invited,no (event_id are unique in the file)
    ## index the event ids in the train data
    ## NOTE: PPL WHO ARE INTERESTED MAY NOT BE INVITED IN THE FIRST PLACE
    ## AND VICE VERSA 
    event_set = set(train['event']) # 8846 events
    attendence_headers = ['event', 
        'event_interests', 'event_potential_interests', 'event_invites', 'event_nointerests',
        'event_interests_ratio', 'event_potential_interests_ratio', 'event_invites_ratio', 'event_nointerests_ratio']
    with open(attendence_path, 'r') as fattendence:
        reader = csv.reader(fattendence)
        reader.next() ## ignore headers reading
        attendence_data = []
        for (event_id, ppl_interest, ppl_maybe, ppl_invited, ppl_notinterest) in reader:
            if int(event_id) in event_set:
                (ppl_interest, ppl_maybe, ppl_invited, ppl_notinterest) = [
                                            ppl_interest.split(' '),
                                            ppl_maybe.split(' '),
                                            ppl_invited.split(' '),
                                            ppl_notinterest.split(' ')]
                (n_interest, n_maybe, n_invited, n_notinterested) = [
                                            len(ppl_interest),
                                            len(ppl_maybe),
                                            len(ppl_invited),
                                            len(ppl_notinterest)]
                n_total = len(set(sum((ppl_interest, ppl_maybe, ppl_invited, ppl_notinterest), []))) * 1.
                attendence_data.append([int(event_id), 
                                        n_interest, n_maybe,
                                        n_invited, n_notinterested,
                                        n_interest/n_total, n_maybe/n_total,
                                        n_invited/n_total, n_notinterested/n_total,])
    attendence = DataFrame(attendence_data, columns = attendence_headers)
    return pd.merge(train, attendence)

    
def fill_train_with_event_clustering(train):
    ## clustering the events based on the common words in their
    ## event_clusters are generated by kmeans (R script cluster_events.R)
    ## generated file has two columns event_id, cluster
    ## each row has a unique event_id
    ## description
    ## fill in the event clustering information in the train data
    event_set = set(train['event']) # 8846 events
    with open(clusters_path, 'r') as fclusters:
        reader = csv.reader(fclusters)
        reader.next() # ignore headers line
        clusters_data = [
            [int(float(event_id)), "topic"+cluster]
            for (event_id, cluster) in reader
            if int(float(event_id)) in event_set
        ]
    clusters = DataFrame(clusters_data, columns=['event', 'topic'])
    return pd.merge(train, clusters)

def load_user_friends(train):
    ## build user_id indexing from train
    user_set = set(train['user']) ## 2034 users in train
    with open(friends_path, 'r') as ffriends:
        reader = csv.reader(ffriends)
        reader.next() ## ignore header
        ## user friends - user him/herself and his/her friend list
        user_friends = dict([[int(user), set([int(user)]+[int(f) for f in friends.split(' ')])] 
                            for (user, friends) in reader
                            if int(user) in user_set])
    return user_friends
    
def load_event_attendees(train):
    ## build event_id index
    def toints(intstr):
        return [] if not intstr else map(int, intstr.split(' '))

    event_set = set(train['event'])
    with open(attendence_path, 'r') as fattendence:
        reader = csv.reader(fattendence)
        reader.next() ## ignore headers reading
        headers = ['ppl_interest', 'ppl_maybe', 'ppl_invited', 'ppl_notinterest']
        event_attendees = {}
        for (event, ppl_interest, ppl_maybe, ppl_invited, ppl_notinterest) in reader:
            if int(event) in event_set:
                event_attendees[int(event)] = dict(zip(
                    headers,
                    [toints(ppl_interest), 
                    toints(ppl_maybe), 
                    toints(ppl_invited), 
                    toints(ppl_notinterest)]
                ))

    return event_attendees
    
def fill_friend_attendees(train, user_friends, event_attendees):
    attendee_headers = ['user', 'event', 'interested_frnds', 'maybe_frnds', 'invited_frnds', 'notinterested_frnds']
    train['interested_frnds'] = train.apply(
    lambda r: len(set(event_attendees[r['event']]['ppl_interest']).intersection(set(user_friends[r['user']]))),
    axis = 1
    )
    train['maybe_frnds'] = train.apply(
    lambda r: len(set(event_attendees[r['event']]['ppl_maybe']).intersection(set(user_friends[r['user']]))),
    axis = 1
    )
    train['invited_frnds'] = train.apply(
    lambda r: len(set(event_attendees[r['event']]['ppl_invited']).intersection(set(user_friends[r['user']]))),
    axis = 1
    )
    train['notinterested_frnds'] = train.apply(
    lambda r: len(set(event_attendees[r['event']]['ppl_notinterest']).intersection(set(user_friends[r['user']]))),
    axis = 1
    )
    return train
    """
    attendee_data = []
    for (i, uid, eid) in train[['user', 'event']].itertuples():
        friends = set(user_friends[uid])
        interested_frnds = len(set(event_attendees[eid]['ppl_interest']).intersection(friends))
        maybe_frnds = len(set(event_attendees[eid]['ppl_maybe']).intersection(friends))
        invited_frnds = len(set(event_attendees[eid]['ppl_invited']).intersection(friends))
        notinterested_frnds = len(set(event_attendees[eid]['ppl_notinterest']).intersection(friends))
        attendee_data.append([uid, eid, interested_frnds, maybe_frnds, invited_frnds, notinterested_frnds])
    attendee_friends = DataFrame(attendee_data, columns=attendee_headers)
    print 'DEBUG: attendee_friends len: ', len(attendee_friends)
    return pd.merge(train, attendee_friends, on = ['user', 'event'], how = 'right')
    """
    
def fill_friend_attendee_ratios(train, user_friends, event_attendees):
    attendee_headers = ['user', 'event', 'interested_frnds_ratio', 
        'maybe_frnds_ratio', 'invited_frnds_ratio', 'notinterested_frnds_ratio']
    train['interested_frnds_ratio'] = train.apply(
    lambda r:  r['interested_frnds'] * 1./ len(user_friends[r['user']]),
    axis = 1
    )
    train['maybe_frnds_ratio'] = train.apply(
    lambda r:  r['maybe_frnds'] * 1./ len(user_friends[r['user']]),
    axis = 1
    )
    train['invited_frnds_ratio'] = train.apply(
    lambda r:  r['invited_frnds'] * 1./ len(user_friends[r['user']]),
    axis = 1
    )
    train['notinterested_frnds_ratio'] = train.apply(
    lambda r:  r['notinterested_frnds'] * 1./ len(user_friends[r['user']]),
    axis = 1
    )
    return train
    
def post_process(train, is_train):
    ## 1. add the friend_with_creator field based on user and creator
    ## load user_friends data
    user_friends = load_user_friends(train)    
    train['friend_with_creator'] = train.apply(
                                lambda r: int(r['event_creator']) in user_friends[r['user']], 
                                axis = 1)
    print 'finish adding friend_with_creator', len(train)
    
    ## 2. add friends_interested, friends_maybe, friends_not_interested
    ## and their ratios over all friends
    event_attendees = load_event_attendees(train)
    train = fill_friend_attendees(train, user_friends, event_attendees)
    print 'finish adding friends_xx', len(train)
    train = fill_friend_attendee_ratios(train, user_friends, event_attendees)
    print 'finish adding friends_xx ratios', len(train)
    
                                
    ## 3. normalize missing values - fill all the ' ' empty values with 'NA'
    train = train.apply(
                lambda r: [(normalized_na if e in na_values else e) for e in r], 
                axis = 1)
                
    ## 4. notification ahead of event
    ## notification time is not available in private test
    
    train['notification_ahead_hrs'] = train.apply(
                                lambda r: (parse(r['event_start_time']) - parse(r['timestamp'])).total_seconds() / 3600., 
                                axis = 1)
    
    ## 5. age
    train['user_age'] = train.apply(
                                lambda r: 2013 - int(r['user_birthyear']) if r['user_birthyear'] is not normalized_na else normalized_na, 
                                axis = 1)
    
    ## 6. user location -- factor too many levels
    ## event_city, event_country, and user_local are relatively clean
    ## user_location contains a string (not well formatted)
    ## create new fields - 
    ## user_in_event_city, user_in_event_country, ??-user_in_event_lang(uage)
    train['user_in_event_city'] = train.apply(
        lambda r: r['event_city'].lower() in r['user_location'].lower(),
        axis = 1
    )
    train['user_in_event_country'] = train.apply(
        lambda r: r['event_country'].lower() in r['user_location'].lower(),
        axis = 1
    )

    
    ## 7. output ranking - ranking or regression - only for train data
    ## using invited and interested - is it a output leakage??
    ## notinterested1_invited1(_interested0) =>1 11
    ## notinterested1_invited0(_interested0) =>2 503
    ## notinterested0_interested0_invited1 =>3 558
    ## notinterested0_interested0_invited0 =>4 10573
    ## (notinterested0)_interested1_invited1 =>5 125
    ## (notinterested0)_interested1_invited0 =>6 4006
    print 'DEBUG: is_train=', is_train
    if is_train:
        def get_rank(row):
            invited, interested, not_interested = row['invited'], row['interested'], row['not_interested']
            return (1 if not_interested==1 and invited==1
                        else 2 if not_interested==1 and invited==0
                        else 3 if not_interested==0 and interested==0 and invited==1
                        else 4 if not_interested==0 and interested==0 and invited==0
                        else 5 if interested==1 and invited==1
                        else 6) - row['event_interests_ratio'] #+ row['event_nointerests_ratio']
        train['interest_rank'] = train.apply(get_rank, axis = 1)
    
    
    ## 8. select the most significant features
    ## select the useful features only
    inputs_in_use = ['user', 'event',
                    ## invited is an output leakage - but it proves useful 
                    ## invited is not availabe in private test data anymore
                    'invited', 
                    'user_locale',
                    'user_in_event_city', 'user_in_event_country',
                    'user_gender', 'friend_with_creator',
                    'event_interests', 'event_potential_interests',
                    'event_invites', 'event_nointerests',
                    'event_interests_ratio', 'event_potential_interests_ratio', 
                    'event_invites_ratio', 'event_nointerests_ratio',
                    ## notification_ahead_hrs is not available in private test anymore
                    'notification_ahead_hrs', 
                    'user_age',
                    'interested_frnds', 'maybe_frnds', 'invited_frnds', 'notinterested_frnds',
                    'interested_frnds_ratio', 'maybe_frnds_ratio', 
                    'invited_frnds_ratio', 'notinterested_frnds_ratio', 
                    'topic', 'user_community',]
    #outputs_in_use = ['interested', 'not_interested', 'interest_rank']
    outputs_in_use = ['interest_rank'] if is_train else []
    train = train[inputs_in_use+outputs_in_use]
    return train
    
def data_to_file(train, fpath):
    train = train.sort('user')
    train.to_csv(fpath, na_rep=normalized_na, header=True, index=False)
     

def main():
    ## load data
    ## tricks - the generated file size should be about the same
    ## as train.csv, other files (specially events.csv) are usually
    ## much larger than train.csv
    ## load train.csv as a dataframe, use csv file reader for others
    
    ## read original and dest file path from command line
    if len(sys.argv) is not 4:
        print 'Usage: %s original_file dest_file train|test' % (sys.argv[0], )
        sys.exit(-1)
    original_path, dest_path, is_train = sys.argv[1:]
    is_train = True if (is_train == 'train') else False
    print 'generate a ', ('TRAIN' if is_train else 'TEST'), 'file...'
    
    ## load original data
    train = load_train_csv(original_path)
    print 'finished reading original data', len(train)
    
    ## fill with the events information first
    ## as filling user information with need the 
    ## creator information later
    train = fill_train_with_events(train)
    print 'finished filling events information', len(train)
    
    ## fill train data with user information
    train = fill_train_with_users(train)
    print 'finished filling user information', len(train)
    
    ## fill train data with user community
    train = fill_train_with_communities(train)
    print 'finished filling user communities', len(train)
    
    ## fill train data with attendence (event popularity?) information
    train = fill_train_with_attendence(train)
    print 'finished filling attendence information', len(train)
    
    ## fill train data with event clustering (based on common words) information
    train = fill_train_with_event_clustering(train)
    print 'finished filling event clustering information', len(train)
    
    ## cleansing and creating new fields
    train = post_process(train, is_train)
    print 'finished post-processing train data'
    
    ## write out the new train data to dest file
    data_to_file(train, dest_path)
    print 'finished writing data to ', dest_path
    

if __name__ == '__main__':
    main()
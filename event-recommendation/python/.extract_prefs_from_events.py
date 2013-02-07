## extract from events.csv information to "eventsbased_user_events_pref.csv (creator, events, 1)"

import os, csv

print 'cutting filed event_id user_id from events.csv'
os.system('cat ../data/events.csv | cut -d , -f 1,2 > ../data/event_creator.csv')
print 'generated event_creator.csv'


reader = csv.reader(open('../data/event_creator.csv', 'r'))
writer = csv.writer(open('../data/eventsbased_user_events_pref.csv', 'w'))

reader.next() ## ignore the header
for (event, user) in reader:
    writer.writerow([user, event, 1])
    
print 'generated ../data/eventsbased_user_events_pref.csv'

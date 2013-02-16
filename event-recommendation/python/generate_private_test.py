## FOR private leaderboard submissions
## convert event_popularity_benchmark_private_test_only.csv
## to test_private.csv (as in the same format of test.csv, 
## except that only 'user' and 'event' columns are in the file)

import csv

def main():
    private_benchmark_path = '../data/event_popularity_benchmark_private_test_only.csv'
    private_test_path = '../data/test_private.csv'
    header = ['user', 'event']
    reader = csv.reader(open(private_benchmark_path, 'r'))
    writer = csv.writer(open(private_test_path, 'w'))
    reader.next()
    writer.writerow(header)
    for (user, events) in reader:
        events = [e[:-1] for e in events[1:-1].split(', ')] # get rid of suffix L (for R integer)
        for event in events:
            writer.writerow([user, event])
    
if __name__ == '__main__':
    main()
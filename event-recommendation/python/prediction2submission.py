## convert test prediction file "../data/test_prediction.csv" to 
## submission-ready format, which is a dictionary of 
## {user_id: [sorted_event_id...]}

import pandas as pd
import sys
from collections import defaultdict


def main():
    if len(sys.argv) != 2:
        print 'Usage: %s submission.file' % (sys.argv[0],)
        sys.exit(-1)
    prediction_file = '../data/test_predictions.csv'
    submission_file = sys.argv[1]
    data = pd.read_csv(prediction_file, header = 0, na_values = ['NA']) 
    user_events = defaultdict(list)
    sorted_df = (data[['user', 'event', 'interest_rank']]
                .sort(['user', 'interest_rank'], ascending=[1, 0]))
    for (uid, eid, interest) in sorted_df.to_records(index=False):
        user_events[uid].append(eid)
    ## write out solution
    users = sorted(user_events.keys())
    events = [' '.join(map(str, user_events[u])) for u in users]
    submission = pd.DataFrame({"User": users, "Events": events})
    submission[["User", "Events"]].to_csv(submission_file, index=False)

if __name__ == '__main__':
    main()
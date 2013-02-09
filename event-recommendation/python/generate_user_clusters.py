## generate a matrix/dataframe of users for clustering
## the matrix is nusers x nusers, each row is a vector of friends (0 -non friends, 1 - friends)

import sys
import pandas as pd
import csv
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix

def main():
    users_path = '../data/user_ids.csv'
    friends_path = '../data/user_friends.csv'
    user_clusters_path = '../data/user_clusters.csv'
    n_clusters = 40 ## for KMeans
    n_components = 500 ## for PCA
    
    ## load user ids
    ## users in user_ids.csv are all unique
    print 'load user ids...'
    with open(users_path, 'r') as fusers:
        reader = csv.reader(fusers)
        reader.next() ## ignore the headers
        user_ids = [int(line[0]) for line in reader]
        
    ## build user - position index
    nusers = len(user_ids)
    user_positions = dict([(user, i) for (i, user) in enumerate(user_ids)])
    
    ## build matrix
    relationship = np.zeros((nusers, nusers), dtype=np.int8)
    #relationship.fill(False)
    
    ## load user friends
    print 'load user friends...'
    user_friend_pos = defaultdict(set)
    with open(friends_path, 'r') as ffriends:
        reader = csv.reader(ffriends)
        reader.next() ## ignore the headers
        for (user, friends) in reader:
            user_row = user_positions[int(user)]
            friends = map(int, friends.split(' ')) if friends else []
            friend_cols = [user_positions[f] for f in friends 
                            if f in user_positions]
            relationship[user_row][friend_cols] = 1
    
    
    ## PCA
    #print 'doing pca with n_compoent =', n_components
    #relationship = PCA(n_components = n_components).fit_transform(relationship)
    
    ## convert matrix to sparse
    ## making sparse matrix
    print 'convert friendship matrix into sparse'
    relationship = coo_matrix(relationship)
    
    ## clustering
    print 'clustering users based on their frienships'
    labels = KMeans(n_clusters = n_clusters,
                        n_jobs=-1).fit_predict(relationship)
                        
    ## write user id and cluster labels
    user_labels = pd.DataFrame({'user':user_ids, 'user_type':labels})
    print user_labels.describe()
    user_labels.to_csv(user_clusters_path, header = True, index = False, )
    
    
if __name__ == '__main__':
    main()
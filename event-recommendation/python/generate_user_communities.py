## generate a network of users based on their 
## friendship, try to find community by best partition

import csv
import networkx as nx
import pickle
import community
import pandas as pd
import itertools

def force_partition(partitions, n_communities):
    ## group users by their community in partitions
    keyfn = lambda (user, community): community
    sorted_partitions = sorted(partitions.items(), key = keyfn)
    community_groups = [ (community, map(lambda (u,c): u,list(users)))
            for (community, users) 
            in itertools.groupby(sorted_partitions, keyfn)]
    major_groups = sorted(community_groups, 
                        key = lambda (c, users): len(users),
                        reverse = True)
    major_groups = major_groups[:(n_communities-1)]
    ## by default - all users are in n_community
    forced_partitions = dict(zip(partitions.keys(),
                    itertools.repeat('level0')))
    ## major groups start with 1 .. n_communities - 1
    for (igroup, (_, users)) in enumerate(major_groups, 1):
        for user in users:
            forced_partitions[user] = 'level'+str(igroup)
    return forced_partitions

def main():
    users_path = '../data/user_ids.csv'
    friends_path = '../data/user_friends.csv'
    graph_path = '../data/user_graph.csv'
    user_communities_path = '../data/user_communities.csv'
    
    n_communities = 31

    ## build the friendship graph
    relationship = nx.Graph()
    
    
    ## load user ids
    ## users in user_ids.csv are all unique
    print 'load user ids...'
    with open(users_path, 'r') as fusers:
        reader = csv.reader(fusers)
        reader.next() ## ignore the headers
        user_ids = [int(line[0]) for line in reader]
    relationship.add_nodes_from(user_ids)
    
    user_set = set(user_ids)
    print '!DEBUG: len of user_set', len(user_set)
    ## load user friends
    print 'load user friends...'
    with open(friends_path, 'r') as ffriends:
        reader = csv.reader(ffriends)
        reader.next() ## ignore the headers
        for (user, friends) in reader:
            user = int(user)
            friends = map(int, friends.split(' ')) if friends else []
            friends = filter(lambda f: f in user_set, friends)
            relationship.add_edges_from((user, f) for f in friends)
    
    print '!DEBUG: len of relationship nodes', len(relationship.nodes())
    ## write built graph to pickle
    pickle.dump(relationship, open(graph_path, 'w'))
    
    ## find communities based on partition
    ## partitions: {uid: community, ....}
    print 'finding community'
    partitions = community.best_partition(relationship)
    print '!DEBUG: len of partition.keys()', len(partitions.keys()) 
    
    ## force the number of communities to be n_communities (within 32 for R)
    print 'force number of communities to', n_communities
    forced_partitions = force_partition(partitions, n_communities)
    
    
    ## write partition data to file
    print 'write user partition to file', user_communities_path
    user_partitions = pd.DataFrame({'user': user_ids, 
                    'user_community': map(lambda u: forced_partitions[u], user_ids)})
    user_partitions.to_csv(user_communities_path, header = True, index = False)
    

                    
    
if __name__ == '__main__':
    main()
## use greedy search and 2-opt to improve the optimality of twin path
## reasonability - most of the twin solution is generated based on grid-searching (in this folder)
## using 2-opt can help to improve by exploring further connected paths
## GREEDY-2-OPT algorithm:
## 1. for each edge of a-b-c-d, exchange b-c to a-c-b-d, calculate the distance delta d(acbd)-d(abcd)
## 2. constrain the above exchange to make sure that a-c, b-d is not repeated in 1st tsp solution
## 3. find the minimum change that satisifying 1, and 2 - do it iteratively

import sys
import pandas as pd
from numpy.linalg import norm

cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
cities = cities.as_matrix()

## buidl distance function
dist_matrix = {}
def get_dist(citya, cityb):
    if citya > cityb: cityb, citya = citya, cityb
    if (citya, cityb) not in dist_matrix:
        dist_matrix[(citya, cityb)] = norm(cities[citya]-cities[cityb])
    return dist_matrix[(citya, cityb)]

def exchangeopt2(pair, path ):
    i, j = pair
    #print i, j
    ## dist(i-1, j, i, j+1) - dist(i-1, i, j, j+1)
    gain = (get_dist(path[i-1], path[j]) + get_dist(path[i], path[j+1]) 
            - get_dist(path[i-1], path[i]) - get_dist(path[j], path[j+1]))
    return (gain, (i, j))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python %s tsp_solution.file twin_solution.file improved_twin.file' % (sys.argv[0], )
    ## load data
    tsp_path = [int(line.strip())-1 for line in open(sys.argv[1]).readlines()]
    twin_path = [int(line.strip())-1 for line in open(sys.argv[2]).readlines()]
    ## set of edges in tsp_path
    tsp_edges = set((tsp_path[i-1], tsp_path[i]) for i in xrange(len(tsp_path)))
    tsp_edges.update((tsp_path[i], tsp_path[i-1]) for i in xrange(len(tsp_path)))
    ## do the local optimization iteratively
    MAX_ITER = 10000
    accumulative_gain = 0
    for it in xrange(MAX_ITER):
        candidates = [(gain, exchange) 
                            for (gain, exchange) 
                                in map(lambda i: exchangeopt2((i, i+1), twin_path), 
                                        [i for i in range(1, len(twin_path)-2) if (twin_path[i-1], twin_path[i+1]) not in tsp_edges 
                                                                            if (twin_path[i], twin_path[i+2]) not in tsp_edges])
                            if gain < 0]
        #print 'candidates: ', type(candidates), len(candidates)
        if not candidates: 
            print 'no improvement anymore after', it, 'iterations'
            break
        (gain, (i, j)) = min(candidates)
        accumulative_gain += gain
        #print 'exchange', (i, j), 'with gain', gain
        if it % 1000 == 0:
            print 'iteration: ', it, 'current total gain, ', accumulative_gain
        twin_path = twin_path[:i] + [twin_path[j], twin_path[i]] + twin_path[j+1:]
    #save improved twin path to the file
    open(sys.argv[3], 'w').write('\r\n'.join([str(city + 1) for city in twin_path]))
    
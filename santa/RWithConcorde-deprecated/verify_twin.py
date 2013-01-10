## verify if two paths are twins, i.e., no edges (a->b b->a either way) in tour1 appears in tour2

import sys
import pandas as pd
import numpy as np
cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
cities = cities.as_matrix()
## euclidean distance
def edist(pt12):
	xy1, xy2 = cities[pt12,:]
	return np.linalg.norm(np.array(xy1)-np.array(xy2))

## euclidean distance of path
def path_dist(path):
	return sum(map(edist, [path[i:i+2] for i in xrange(len(path)-1)]))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s tsp_solution.file twin_solution.file' % (sys.argv[0], )
    ## load data
    tsp_path = [int(line.strip())-1 for line in open(sys.argv[1]).readlines()]
    twin_path = [int(line.strip())-1 for line in open(sys.argv[2]).readlines()]
    ## get distance of each path
    print 'total distance of tsp path: ', path_dist(tsp_path)
    print 'total distance of twin path: ', path_dist(twin_path)
    print 'verify lengths match'
    assert len(tsp_path) == len(twin_path)
    print 'verify cities match'
    assert set(tsp_path) == set(twin_path) == set(range(150000))
    print 'verify no duplicate edges (biway)'
    ## set of edges in tsp_path
    tsp_edges = set((tsp_path[i-1], tsp_path[i]) for i in xrange(len(tsp_path)))
    tsp_edges.update((tsp_path[i], tsp_path[i-1]) for i in xrange(len(tsp_path)))
    ## verify
    for edge in [(twin_path[i-1], twin_path[i]) for i in xrange(len(twin_path))]:
        if edge in tsp_edges:
            print edge
            raise Exception('duplicated edge in twin path')
    print 'VERIFY DONE...'
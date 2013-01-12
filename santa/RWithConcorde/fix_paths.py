## verify if two paths are twins, i.e., no edges (a->b b->a either way) in tour1 appears in tour2

import sys


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python %s tsp_solution.file twin_solution.file' % (sys.argv[0], )
    ## load data
    tsp_path = [int(line.strip())-1 for line in open(sys.argv[1]).readlines()]
    twin_path = [int(line.strip())-1 for line in open(sys.argv[2]).readlines()]
    ## set of edges in tsp_path
    tsp_edges = set((tsp_path[i-1], tsp_path[i]) for i in xrange(len(tsp_path)))
    tsp_edges.update((tsp_path[i], tsp_path[i-1]) for i in xrange(len(tsp_path)))
    ## very simple fixing in (a, b, c), (a, b) is duplicated, exchange b, c -> (a, c, b)
    total_fixed = 0
    offset = 1
    for i in xrange(len(twin_path)-2):
        if (twin_path[i], twin_path[i+1]) in tsp_edges:
            total_fixed += 1
            #print i, i+1+offset
            twin_path[i+1], twin_path[i+1+offset] = twin_path[i+1+offset], twin_path[i+1]
    ## save twin path solution
    open(sys.argv[2], 'w').write('\r\n'.join([str(city + 1) for city in twin_path]))
    ## get distance of each path
    print 'fix DONE with %d fixes...' % (total_fixed, )
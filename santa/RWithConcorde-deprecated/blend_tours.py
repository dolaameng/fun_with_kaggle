## blend two tours (tsp and twin) to make their performance closer to the average
## ALGORITHM: tsp tour and twin are partitioned into chunks, each chunk on both tours
## have the exact same set of points, besides, there are no CONFLICTS within the chunks 
## between tsp and twin tour
## 1. break tsp and twin tours into chunks, connect them chunk by chunk, exchange of chunks could happen
## 2. exchange a chunk when the accumulative diff between tsp and twin dists and current diff are of the same signs
## e.g., tsp tour -> t1, t2, t3, t4, t5
##      twin tour -> w1, w2, w3, w4, w5
## when connecting to chunk 2, if global_diff(t1, w1) is negative, but current_diff(t2, w2) is positive, 
## then it should be (t1, t2) and (w1, w2)
## when connecting to chunk 2, if global_diff(t1, w1) is negative, but current_diff(t2, w2) is negative, 
## then it should be (t1, w2) and (w1, t2)

import sys
import numpy as np
import pandas as pd

## city xy coordinates
cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
cities = cities.as_matrix()

## euclidean distance
def edist(pt12):
	xy1, xy2 = cities[pt12,:]
	return np.linalg.norm(np.array(xy1)-np.array(xy2))

## euclidean distance of path
def path_dist(path):
	return sum(map(edist, [path[i:i+2] for i in xrange(len(path)-1)]))

def check_same_chunks(tsp_path, twin_path, chunk_sizes):
    for ichunk in xrange(len(chunk_sizes)):
        begin = 0 if ichunk == 0 else sum(chunk_sizes[:ichunk])
        end = begin + chunk_sizes[ichunk]
        #print 'ichunk = %d, begin = %d, end = %d, chunksize = %d' % (ichunk, begin, end, chunk_sizes[ichunk])
        if set(tsp_path[begin:end]) != set(twin_path[begin:end]):
            print 'ichunk = %d, begin = %d, end = %d, chunksize = %d' % (ichunk, begin, end, chunk_sizes[ichunk])
            print 'difference of tsp - twin', set(tsp_path[begin:end]) - set(twin_path[begin:end])
            raise Exception('chunk in tsp and twin does not match')
    print 'finished checking chunk match...'

def dist_diff(path1, path2):
    return path_dist(path1) - path_dist(path2)
def sign(n):
    return +1 if n >= 0 else -1

if __name__ == '__main__':
    if len(sys.argv) != 6:
        print 'Usage: python %s tsp_solution.file twin_solution.file chunk_sizes.file blended_tsp.file blended_twin.file' % (sys.argv[0], )
    ## load data
    tsp_path = [int(line.strip())-1 for line in open(sys.argv[1]).readlines()]
    twin_path = [int(line.strip())-1 for line in open(sys.argv[2]).readlines()]
    chunk_sizes = [int(line.strip()) for line in open(sys.argv[3]).readlines()]
    ## perform checking of the two solutions to see if they share the same chunk
    check_same_chunks(tsp_path, twin_path, chunk_sizes)
    blended_tsp, blended_twin = tsp_path[:chunk_sizes[0]], twin_path[:chunk_sizes[0]]
    accumulative_tsp2twin = dist_diff(blended_tsp, blended_twin)
    for ichunk in xrange(1, len(chunk_sizes)):
        begin = sum(chunk_sizes[:ichunk])
        end = begin + chunk_sizes[ichunk]
        tsp_chunk = tsp_path[begin:end]
        twin_chunk = twin_path[begin:end]
        current_tsp2twin = dist_diff(tsp_chunk, twin_chunk)
        ## cross
        if sign(accumulative_tsp2twin) == sign(current_tsp2twin):
            blended_tsp += twin_chunk
            blended_twin += tsp_chunk
            accumulative_tsp2twin -= current_tsp2twin
        ## remain
        else:
            blended_tsp += tsp_chunk
            blended_twin += twin_chunk
            accumulative_tsp2twin += current_tsp2twin
    ## perform checking on two blended
    check_same_chunks(blended_tsp, blended_twin, chunk_sizes)
    ## write blended new solution
    open(sys.argv[4], 'w').write('\r\n'.join([str(city + 1) for city in blended_tsp]))
    open(sys.argv[5], 'w').write('\r\n'.join([str(city + 1) for city in blended_twin]))
            
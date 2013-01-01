## find the greedy search solution for santa tsp
## algorithm:
## 1. with a starting point, find the nearst neighbor from
## the prebuilt neighbor list,
## which has no conflicts (has not seen before on the path)
## 2. if we cannot find any from the prebuilt neighbor list
## for now we will randomly pick one from the unassigned list

import pickle
import sys
import numpy as np
import pandas as pd

## load [[neighbors for 0], [neighbors for i], ... [neighbors for n]] structure
neighbors = pickle.load(open('../data/neighbors500.pickled'))
print 'finish loading neighbors structure'
## city xy coordinates
cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
cities = cities.as_matrix()
ncity = len(neighbors)
print 'finish loading city xys'
## pre-sort cities based on their x and y s
sorted_city_xs = sorted(xrange(ncity), key = lambda c: cities[c][0])
sorted_city_ys = sorted(xrange(ncity), key = lambda c: cities[c][1])
print 'finish sorting on x and y'

## visited and unvisited cities

allcities = set(xrange(ncity)) # 0 ... ncity-1
visited = set()
unvisited = allcities

## euclidean distance
def edist(pt12):
	xy1, xy2 = cities[pt12,:]
	return np.linalg.norm(np.array(xy1)-np.array(xy2))

## euclidean distance of path
def path_dist(path):
	return sum(map(edist, [path[i:i+2] for i in xrange(len(path)-1)]))

## find the approximately closest neighbor
def find_closet(pt, candidates):
	city, dist = min(map(lambda c: (c, abs(cities[c][0]-cities[pt][0])), candidates), key = lambda (c, d): d)
	return city

## axis should be 'x' or 'y'
def find_closet_on_xy(pt, candidates, axis = 'x'):
	sorted_city = sorted_city_xs if axis == 'x' else sorted_city_ys
	pt_index = sorted_city.index(pt)
	left, right = pt_index-1, pt_index+1
	while left >= 0 or right <= len(sorted_city)-1:
		if left >= 0 and sorted_city[left] in candidates:
			return sorted_city[left]
		elif right <= len(sorted_city)-1 and sorted_city[right] in candidates:
			return sorted_city[right]
		else:
			left, right = left-1, right+1
	raise Exception('cannot find closet neighbor')

profile = True
if __name__ == '__main__':
	start = ncity / 2
	path = [start]
	visited.add(start)
	unvisited.remove(start)
	missed = 0
	## clusters for profiling purpose
	if profile:
	    cluster = [start]
	    clusters = []
	while unvisited:
		current = path[-1]
		candidates = [n for n in neighbors[current] if n != current and n not in visited]
		if candidates:
			nextnode = candidates[0]		
		else:
			#nextnode = find_closet(current, unvisited)
			nextnode_onx = find_closet_on_xy(current, unvisited, 'x')
			nextnode_ony = find_closet_on_xy(current, unvisited, 'y')
			dist_onx = edist([current, nextnode_onx])
			dist_ony = edist([current, nextnode_ony])
			nextnode = nextnode_onx if dist_onx < dist_ony else nextnode_ony
			missed += 1
			if profile:
			    clusters.append(cluster)
			    cluster = []
		if profile: cluster.append(nextnode)
		unvisited.remove(nextnode)
		visited.add(nextnode)
		path.append(nextnode)
	if profile: clusters.append(cluster)
	print len(path)
	print path_dist(path)
	print '%d nodes have been missed in greedy search' % (missed, )
	print 'dumping solutions ...'
	pickle.dump(path, open('../solution/greedy-solution.pickled', 'w'))
	if profile:
	    pickle.dump(clusters, open('../solution/greedy-cluster.pickled', 'w'))
	    
	## for neighbor100, the result is:
	##150000
	##10114906.4536
	##1087 nodes have been missed in greedy search
	## for neighbor200, the result is:
	##150000
    ##8614095.72302
    ##546 nodes have been missed in greedy search
    ## for neighbors500
    ##7790257.84424
    ##209 nodes have been missed in greedy search
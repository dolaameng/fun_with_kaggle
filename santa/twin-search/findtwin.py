## using search to find the 2nd path after the 1st path is given
## the constraint is: any partial path (a, b) or (b, a) in both directions in 
## path1 cannot be used in path2
## the distance of path2 should be as short as possible - actually
## we suspect its distance should be close to the 1st one if both are close to
## optimal - specially when the points are distributed in a dense rectangle

## use AIMA search library

from search import *
import pickle
import sys
import numpy as np
import pandas as pd
from bitarray import bitarray

## load global data of city maps
## load [[neighbors for 0], [neighbors for i], ... [neighbors for n]] structure
neighbors = pickle.load(open('../data/sparse_neighbors300.pickled'))
print 'finish loading neighbors structure'
## city xy coordinates
cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
cities = cities.as_matrix()
ncity = len(neighbors)
#allcities = set(xrange(ncity)) # 0 ... ncity-1
print 'finish loading city xys'
## pre-sort cities based on their x and y s
sorted_city_xs = sorted(xrange(ncity), key = lambda c: cities[c][0])
sorted_city_ys = sorted(xrange(ncity), key = lambda c: cities[c][1])
sorted_city_xs_index = dict((v, k) for (k, v) in enumerate(sorted_city_xs))
sorted_city_ys_index = dict((v, k) for (k, v) in enumerate(sorted_city_ys))
print 'finish sorting on x and y and sorting index'

## load ref_path
ref_path = pickle.load(open('../solution/greedy-solution.pickled'))
print 'load ref_path done'

## euclidean distance of path
def path_dist(path):
	return sum(map(edist, [path[i:i+2] for i in xrange(len(path)-1)]))

## euclidean distance
def edist(city12):
    xy1, xy2 = cities[city12,]
    return np.linalg.norm(np.array(xy1)-np.array(xy2))


## axis should be 'x' or 'y'
def find_closet_on_xy(pt, candidates, axis = 'x'):
    sorted_city = sorted_city_xs if axis == 'x' else sorted_city_ys
    sorted_city_index = sorted_city_xs_index if axis == 'x' else sorted_city_ys_index
    pt_index = sorted_city_index[pt]
    left, right = pt_index-1, pt_index+1
    while left >= 0 or right <= len(sorted_city)-1:
        if left >= 0 and candidates[sorted_city[left]]:
            return sorted_city[left]
        elif right <= len(sorted_city)-1 and candidates[sorted_city[right]]:
            return sorted_city[right]
        else:
            left, right = left-1, right+1
    print pt, axis, pt_index, [c for c in xrange(len(candidates)) if candidates[c]], '!!!'
    raise Exception('cannot find closet neighbor')

## define the search problem
## STATE: (currentnode, bitlist of cities visited so far (partial path))
## ACTION: next city to visit
class TSPTwinProblem(Problem):
    def __init__(self, initial, ref_path, goal = None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        ## goal is the total distance of the ref_path
        ## ref_path is a list of cities - the first path which plays as constraint
        ## for our solution
        ## initial is the starting node - the starting city of the path
        self.initial = initial
        self.ref_path = ref_path
        self.missed = 0
        self.ref_path_dist = goal
        ## build path neighbors
        self.ncity = len(ref_path)
        self.path_neighbors = dict([(ref_path[ic], (ref_path[ic-1], ref_path[(ic+1)%self.ncity])) 
                                    for ic in ref_path])
        self.pathcost = 0
        print 'initalization of tsp done'
        
    def copy_invert(self, bits, i):
        newbits = bits.copy()
        newbits[i] = not newbits[i]
        return newbits
    #@profile
    def successor(self, state):
        """Given a state, return a sequence of (action, state) pairs reachable
        from this state. If there are many successors, consider an iterator
        that yields the successors one at a time, rather than building them
        all at once. Iterators will work fine within the framework."""
        ## finding the successors based on the nearest neighbors
        current, visited = state
        #unvisited = bitarray(self.ncity)
        #unvisited.setall(True)
        #for c in state:
            #unvisited[c] = False
        unvisited = visited.copy()
        unvisited.invert()
        left, right = self.path_neighbors[current]
        unvisited[left] = False
        unvisited[right] = False

        candidates = [next([n for n in neighbors[current] if n != 150000 and unvisited[n]])]
        if not candidates:
            nextnode_onx = find_closet_on_xy(current, unvisited, 'x')
            nextnode_ony = find_closet_on_xy(current, unvisited, 'y')
            self.missed += 1
            candidates = [nextnode_onx, nextnode_ony]
            #candidates = [c for c in range(ncity) if unvisited[c]][:3]
        return ((action, (action, self.copy_invert(visited, action))) for action in candidates)
    def goal_test(self, state):
        current, visited = state
        if visited.count(True) % 10000 == 0:
            print visited.count(True)
        return visited.count(True) == 50000
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        node1, _ = state1
        node2, _ = state2
        self.pathcost = c + edist([node1, node2])
        return self.pathcost

    def value(self):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        return -self.pathcost

    def h(self, node):
        ## heuristic function for several method such as A* search
        state = node.state
        return self.ref_path_dist - self.pathcost
#@profile
def main():
    #ref_path = range(ncity)
    ref_path_dist = path_dist(ref_path)
    start, visited = (ref_path[0], bitarray(150000))
    visited.setall(False)
    visited[ref_path[0]] = True
    initial = (start, visited)
    tsp = TSPTwinProblem(initial, ref_path, goal = ref_path_dist)
    #solution = depth_first_tree_search(tsp)
    solution = astar_search(tsp)
    print 'search done...'
    print len(solution.path())
    print 'missed times:', tsp.missed
    print 'path cost:', tsp.pathcost
    twin_path = map(lambda node: node.state[0], solution.path())
    twin_path.reverse()
    print 'fetching solution done...'
    #print len(twin_path), len(set(twin_path))
    
    twin_path_dist = path_dist(twin_path)
    print ref_path_dist, twin_path_dist
    pickle.dump(twin_path, open('../solution/twin-solution.pickled', 'w'))

if __name__ == '__main__':
    main()
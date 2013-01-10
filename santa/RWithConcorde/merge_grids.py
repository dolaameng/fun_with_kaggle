## the tsp problem is solved by partitioning the whole space into
## grids of fixed sizes. The solutions to each grids can be merged
## in a relatively optimal way to generate better solutions than
## naively connecting the end of a previous grid to the start of 
## the next grid.
## Merging algorithm:
## CONNECT GRIDS ONE BY ONE (pairs of grid) -> (((g1, g2), g3), g4) ... 
## for each grid pair g1 g2:
## rotate g1 and g2 to make g1.end and g2.end minimized

import numpy as np
import pandas as pd
import sys

## load city xys
cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)

def load_subpaths(tour_csv, chunks_csv):
    ## modify 1-index to 0-index city ids
    tour = map(lambda line: int(line.strip())-1, open(tour_csv).readlines())
    chunk_sizes = map(lambda line: int(line.strip()), open(chunks_csv).readlines())
    assert len(tour) == sum(chunk_sizes)
    subindices = np.cumsum(np.array([0]+chunk_sizes))
    subtours = map(lambda i: tour[subindices[i]:subindices[i+1]], range(len(subindices)-1))
    assert len(subtours) == len(chunk_sizes)
    assert sum(map(len, subtours)) == len(tour)
    assert set(sum(subtours, [])) == set(tour)
    return subtours

def rotate_ends(tour, pivot):
    ipivot = tour.index(pivot)
    assert len(tour) > ipivot >= 0
    return tour[ipivot+1:] + tour[:ipivot+1]
    
def rotate_starts(tour, pivot):
    ipivot = tour.index(pivot)
    assert len(tour) > ipivot >= 0
    return tour[ipivot:] + tour[:ipivot]
    
def merge(sofar_tour, itour, subtours):
    if not sofar_tour:
        return subtours[itour]
    tour1, tour2 = subtours[itour-1], subtours[itour]
    ## find the boundary of each tour grids lower_x, upper_x, lower_y, upper_y
    grid1, grid2 = cities.ix[tour1], cities.ix[tour2]
    (x1min, x1max), (y1min, y1max) = (grid1['x'].min(), grid1['x'].max()), (grid1['y'].min(), grid1['y'].max())
    (x2min, x2max), (y2min, y2max) = (grid2['x'].min(), grid2['x'].max()), (grid2['y'].min(), grid2['y'].max())
    ## four possible directions -> (right) \|/ (down) <- (left) /|\ (up)
    ## guess the direction 
    #print (x1min, x1max), (y1min, y1max)
    #print (x2min, x2max), (y2min, y2max)
    if x2min >= x1max: ## right
        pivot1 = grid1.index[grid1['x'].argmax()]
        pivot2 = grid2.index[grid2['x'].argmin()]
    elif x1min >= x2max: ## left
        pivot1 = grid1.index[grid1['x'].argmin()]
        pivot2 = grid2.index[grid2['x'].argmax()]
    elif y2min >= y1max: ## up
        pivot1 = grid1.index[grid1['y'].argmax()]
        pivot2 = grid2.index[grid2['y'].argmin()]
    else: ## down
        pivot1 = grid1.index[grid1['y'].argmin()]
        pivot2 = grid2.index[grid2['y'].argmax()]
    return rotate_ends(sofar_tour, pivot1) + rotate_starts(tour2, pivot2)
    #print grid1.ix[pivot1]
    #print grid2.ix[pivot2]
    #return sofar_tour + tour2
    

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: %s tour.csv chunk_size.csv improved_tour.csv' % (sys.argv[0], )
        exit(-1)
    subtours = load_subpaths(sys.argv[1], sys.argv[2])
    improved_tours = reduce(lambda sofar_tour, itour: merge(sofar_tour, itour, subtours), range(len(subtours)), [])
    assert len(improved_tours) == sum(map(len, subtours))
    assert set(improved_tours) == set(sum(subtours, []))
    open(sys.argv[3], 'w').write('\r\n'.join(map(lambda n: str(n+1), improved_tours)))
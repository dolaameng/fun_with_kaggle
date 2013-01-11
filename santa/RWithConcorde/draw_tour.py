## draw a single tour on the map

import sys
import pandas as pd
import matplotlib.pylab as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s tour.csv' % (sys.argv[0], )
        exit(-1)
    ## load cities 
    cities = pd.read_csv('../data/santa_cities.csv', header = 0, index_col = 0)
    ## load tours
    tours = map(lambda line: int(line.strip())-1, open(sys.argv[1]).readlines())
    ## draw
    plt.figure()
    tourxys = cities.ix[tours, :]
    plt.plot(tourxys['x'], tourxys['y'], 'g--.')
    plt.show()
## generate csv file submission (path1, path2)

import sys
import pandas as pd
from pandas import DataFrame

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'Usage: python %s tsp_solution.file twin_solution.file submission.file' % (sys.argv[0], )
    ## load data
    tsp_path = [int(line.strip())-1 for line in open(sys.argv[1]).readlines()]
    twin_path = [int(line.strip())-1 for line in open(sys.argv[2]).readlines()]
    solution = DataFrame({'path1': tsp_path, 'path2': twin_path})
    solution.to_csv(sys.argv[3], header = True, index = False)
    print 'done...'
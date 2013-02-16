## the solution for public leaderboard test has been released
## the script merges test_classifier.csv (from test.csv and other files)
## with public_leaderboard_solution.csv

## THE USRE SET in public_leaderboard_solution.csv is a subset of users in test.csv
## because users with duplicate timestamp have been left out.
## For EACH USER in public_leaderboard_solution.csv, there is only ONE event given 
## indicating the users interest.

import pandas as pd

def main():
    solution_path = '../data/public_leaderboard_solution.csv'
    test_path = '../data/test_classifier.csv'
    test_solution_path = '../data/test_classifier_solution.csv'
    ## read public_leaderboard_solution.csv data
    solution = pd.read_csv(solution_path, header = 0)
    solution.columns = ['user', 'event', 'usage'] # normalize the header to test_data
    ## read test data
    test = pd.read_csv(test_path, header = 0)
    ## merge data
    #test_solution = pd.merge(solution, test, how='left', on = ['user', 'event'])
    test_solution = pd.merge(solution, test, how='right', on = ['user', 'event'])
    test_solution['usage'] = test_solution['usage'].fillna('NOINTEREST')
    ## write out
    test_solution.to_csv(test_solution_path, header = True, index = False, na_rep='NA')

if __name__ == '__main__':
    main()
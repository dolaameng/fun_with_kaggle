#! env sh

## get tsptour.csv and twintour.csv from running grid_tsp.R and grid_twin_tsp.R

## verify the original solution -- usually there are confilicts
python verify_twin.py tsptour.csv twintour.csv 

## blend them
python blend_tours.py tsptour.csv twintour.csv chunk_sizes.csv blend_tsp.csv blend_twin.csv

## verify to see the distance gap reduced
python verify_twin.py blend_tsp.csv blend_twin.csv

## optimized in-between-grid distance by rotating the grid tour
python merge_grids.py blend_tsp.csv chunk_sizes.csv tspmerged.csv
python merge_grids.py blend_twin.csv chunk_sizes.csv twinmerged.csv

## verify to see the distance reduction for both
python verify_twin.py tspmerged.csv twinmerged.csv

## fix conflicts -- if some of them persist, FIX THEM MANUALLY
## put twin as 1st and tsp as 2nd -- code changes 2nd file (usually tsp is better than twin, so we want to make worse tsp only)
## do it till no conflicts exist
python fix_paths.py twinmerged.csv tspmerged.csv

## improve each solution by greedy2opt
## generate first improved_twintour.csv
python improve_twin_greedy2opt.py tspmerged.csv twinmerged.csv improved_twintour.csv
## generate first improved_tsptour.csv
python improve_twin_greedy2opt.py improved_twintour.csv tspmerged.csv improved_tsptour.csv
## do it iteratively for each till no change
python improve_twin_greedy2opt.py improved_tsptour.csv improved_twintour.csv improved_twintour.csv
python improve_twin_greedy2opt.py improved_twintour.csv improved_tsptour.csv improved_tsptour.csv

## see the results
python verify_twin.py improved_tsptour.csv improved_twintour.csv

## generate submission
python generate_submission.py improved_tsptour.csv improved_twintour.csv submissionx.csv

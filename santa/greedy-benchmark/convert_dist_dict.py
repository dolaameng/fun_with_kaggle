## convert the dist matrix pickle into a dict of dict

import pickle
## list of tuples (city1, city2, l2dist)
cities = pickle.load(open('../data/distance_matrix_50.pickle'))
## preprocess the dist data into a dict(city1, dict)
citydists = {}
for (c1, c2, d) in cities:
	if c1 not in citydists:
		citydists[c1] = {}
	citydists[c1][c2] = d
## save the build pickle
print pickle.dumps(citydists),
#!/usr/bin/python
'''
Simplest OpenOpt TSP example;
requires networkx (http://networkx.lanl.gov)
and FuncDesigner installed.
For some solvers limitations on time, cputime, "enough" value, basic GUI features are available.
See http://openopt.org/TSP for more details
'''
from openopt import *
from numpy import sin, cos
import networkx as nx
import math

nrows, ncols = 500, 30
N = nrows * ncols
cityxys = [(r, c) for r in range(nrows) for c in range(ncols)]

citydists = {}
def edist(c1, c2):
    if c1 == c2: return 0
    if c1 > c2: c1, c2 = c2, c1
    if (c1, c2) not in citydists:
        x1, y1 = cityxys[c1]
        x2, y2 = cityxys[c2]
        citydists[(c1, c2)] = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return citydists[(c1, c2)]
    


print 'built city xys'

G = nx.Graph()
G.add_edges_from(\
                 [(i,j,{'cost': edist(cityxys[i], cityxys[j])}) for i in range(N) for j in range(N) if i != j and edist(cityxys[i], cityxys[j]) < 2])

print 'buidl city map network'

# default objective is "weight"
# parameter "start" (node identifier, number or string) is optional
p = TSP(G, objective = 'cost', start = 0) #, [optional] returnToStart={True}|False, constraints = ..., etc

r = p.solve('sa') # also you can use some other solvers - sa, interalg, OpenOpt MILP solvers

# if your solver is cplex or interalg, you can provide some stop criterion, 
# e.g. maxTime, maxCPUTime, fEnough etc, for example 
# r = p.solve('cplex', maxTime = 100, fEnough = 10) 
# i.e. stop if solution with 10 nodes has been obtained or 100 sec were elapsed
# (gplk and lpSolve has no iterfcn connected and thus cannot handle those criterions)

# also you can use p.manage() to enable basic GUI (http://openopt.org/OOFrameworkDoc#Solving) 
# it requires tkinter installed, that is included into PythonXY, EPD;
# for linux use [sudo] easy_install tk or [sodo] apt-get install python-tk
#r = p.manage('cplex')

print(r.nodes)
#print(r.edges)
#print(r.Edges)
'''
Solver:   Time Elapsed = 0.05 	CPU Time Elapsed = 0.04 
# (glpk takes 5 sec for N = 15, Intel Atom 1.6 GHz)
objFunValue: 5.1628774 (feasible, MaxResidual = 0)
[2, 0, 4, 1, 3, 2]
[(2, 0), (0, 4), (4, 1), (1, 3), (3, 2)]
[(2, 0, {'cost': 8.8185948536513639, 'time': 0.51132677471086396}), 
(0, 4, {'cost': 17.486395009384143, 'time': 0.17994411205270408}), 
(4, 1, {'cost': 9.5669996211204236, 'time': 2.1164007698022411}), 
(1, 3, {'cost': 6.3628446278560142, 'time': 1.0875234238200695}), 
(3, 2, {'cost': 1.4499463430254496, 'time': 1.26768233210464})]
'''

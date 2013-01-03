## given an existing TSP solution, find a TWIN solution 
## satisfying that any path in the original solution will not appear in the TWIN
## ALGORITHM: 
## 1. for connecting grid - reorder them 
## (e.g., rowsbyrows -> colsbycols OR reverse grid order OR specifying different starting node)
## 2. for within each grid - change the distance matrix to make the existing path entry INF

## COMPARED WITH GRID_TWIN_TSP, the main difference is that the twin solution can use a different
## grid (usually bigger, rougher) from the first solution. This allows more "neighbors" available
## for a node
setwd("~/workspace/fun_with_kaggle/santa/RWithConcorde")
library(TSP)
library(doMC)
registerDoMC(cores=4)

##load the existing grided TSP solution
load('tspgridtour.RData')
## OVERWRITE nrow, ncol HERE!!!!!!!!!!!!
nrow <- 8
ncol <- 8
## load city coordinates
cities <- read.csv('../data/santa_cities.csv', header = T)
## function of making equal-sized grid in the map
make.grids <- function(data.x.y, nrow, ncol, xrange=20000, yrange=20000) {
  x.breaks <- (0:ncol) * xrange / ncol
  y.breaks <- (0:nrow) * yrange / nrow
  x.grids <- as.integer(cut(data.x.y$x, breaks=x.breaks, right=T, include.lowest=T))
  y.grids <- as.integer(cut(data.x.y$y, breaks=y.breaks, right=T, include.lowest=T))
  data.x.y$xregion <- as.factor(x.grids)
  data.x.y$yregion <- as.factor(y.grids)
  return (data.x.y)
}
## redo city grids based on new nrow, ncol
city.grids <- make.grids(cities, nrow = nrow, ncol = ncol)
grid.index <- matrix(nrow = 2, ncol = nrow*ncol)
for (xgrid in 1:ncol) {
  for (ygrid in 1:nrow) {
    grid.index[,(xgrid-1)*ncol + ygrid] <- c(xgrid, ygrid)
  }
}
## FIRST TSP SOLUTION AS REFERENCE
tsp.tour <- c(unlist(grided.tours))
##-- build the tour_orders - row by row (the order of grided.tours)
##--tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
##--tour_orders = as.vector(tour_orders)
## make the distance matrix entry on existing tour VERY LARGE
BIG.NUM <- Inf#100000000 # within 2k x 2k
jam_dist_matrix <- function(dist.matrix){
  city.labels <- labels(dist.matrix)
  dist.matrix <- as.matrix(dist.matrix)
  for(city in city.labels) {
    icity <- which(tsp.tour == city)
    if (icity > 1) {
      precity <- tsp.tour[icity-1]
      if (precity %in% city.labels) {
        dist.matrix[precity, city] <- BIG.NUM
        dist.matrix[city, precity] <- BIG.NUM
      }
    }
    if (icity < length(tsp.tour)) {
      nextcity <- tsp.tour[icity+1]
      if (nextcity %in% city.labels) {
        dist.matrix[nextcity, city] <- BIG.NUM
        dist.matrix[city, nextcity] <- BIG.NUM
      }
    }
  }
  return (as.dist(dist.matrix))
}
## solve TSP for twins
twin.tours <- foreach(xy = grid.index, .packages=c('TSP'), .inorder = F) %dopar% {
  xindex <- xy[1]
  yindex <- xy[2]
  dist.matrix <- with(city.grids, dist(city.grids[xregion==xindex & yregion==yindex, 2:3]))
  dist.matrix <- jam_dist_matrix(dist.matrix)
  tsp <- TSP(dist.matrix)
  tour <- solve_TSP(tsp, method="linkern")
  #tour <- solve_TSP(tsp, method="nn")
  #tour <- solve_TSP(tsp, method="nn", control=list(start = length(grid.tour))) # different start for nn
  list(tsp = tsp, tour = tour)
}

complete_tour <- function(grid.index, tsp.tours, city.grids) {
  tsps <- sapply(tsp.tours, function(p.t){p.t[[1]]})
  tours <- sapply(tsp.tours, function(p.t){p.t[[2]]})
  inner_tours_length = sum(sapply(tsp.tours, function(p.t){tour_length(p.t[[1]], p.t[[2]])}))
  #wrong - tour_orders = sapply(1:nrow, function(r){if (r%%2==1) (r-1)*ncol+seq(1,ncol) else (r-1)*ncol+seq(ncol, 1)})
  tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
  ## use the reverse order for grid
  tour_orders = rev(as.vector(tour_orders))
  ## calculate the outer tour length
  outer_tours_length <- 0
  for(i in 1:(length(tour_orders)-1)) {
    pre_tour <- tours[[tour_orders[i]]]
    next_tour <- tours[[tour_orders[i+1]]]
    pre_end <- tail(labels(pre_tour), n = 1)
    next_start <- head(labels(next_tour), n = 1)
    #print(c(pre_tour, next_tour))
    outer_tours_length <- outer_tours_length + dist(city.grids[c(pre_end, next_start), 2:3])[[1]]
  }
  full.tour <- as.integer(unlist(sapply(tours[tour_orders], function(t){labels(t)})))
  return (list(full.tour = full.tour, full.tour.length = inner_tours_length + outer_tours_length))
}

twin.tour <- complete_tour(grid.index, twin.tours, city.grids)
print (twin.tour$full.tour.length)
print (length(twin.tour$full.tour))

## save the tour to csv
write.table(twin.tour$full.tour, file = './twintour.csv', sep = '\r\n', col.names = F, row.names = F)

## plotting
regioin.colors <- with(city.grids, (as.integer(xregion)-1)*10+as.integer(yregion))
with(city.grids, plot(x, y, pch='.', col=regioin.colors))
with(city.grids, lines(x[twin.tour$full.tour], y[twin.tour$full.tour], col=regioin.colors))
title(paste('twin tour with distance ', twin.tour$full.tour.length))
## given an existing TSP solution, find a TWIN solution 
## satisfying that any path in the original solution will not appear in the TWIN
## ALGORITHM: 
## 1. for connecting grid - reorder them 
## (e.g., rowsbyrows -> colsbycols OR reverse grid order OR specifying different starting node)
## 2. for within each grid - change the distance matrix to make the existing path entry INF
setwd("~/workspace/fun_with_kaggle/santa/RWithConcorde")
library(TSP)
library(doMC)
registerDoMC(cores=4)

##load the existing grided TSP solution
load('tspgridtour.RData')
##-- build the tour_orders - row by row (the order of grided.tours)
##--tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
##--tour_orders = as.vector(tour_orders)
## make the distance matrix entry on existing tour VERY LARGE
BIG.NUM <- Inf#100000000 # within 2k x 2k
jam_dist_matrix <- function(dist.matrix, grid.tour){
  dist.matrix <- as.matrix(dist.matrix)
  stopifnot(nrow(dist.matrix) == length(grid.tour))
  for(i in 1:(length(grid.tour)-1)) {
    city1 <- grid.tour[i]
    city2 <- grid.tour[i+1]
    dist.matrix[city1, city2] <- BIG.NUM
    dist.matrix[city2, city1] <- BIG.NUM
  }
  return (as.dist(dist.matrix))
}
## solve TSP for twins
twin.tours <- foreach(xy = grid.index, .packages=c('TSP'), .inorder = F) %dopar% {
  xindex <- xy[1]
  yindex <- xy[2]
  dist.matrix <- with(city.grids, dist(city.grids[xregion==xindex & yregion==yindex, 2:3]))
  grid.tour <- grided.tours[[(xindex-1) * ncol + yindex]] ## ALWAYS (x-1) * ncol + y
  dist.matrix <- jam_dist_matrix(dist.matrix, grid.tour)
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
  tour_orders <- sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
  ## -- use the reverse order for grid
  ## -- tour_orders = rev(as.vector(tour_orders))
  ## DONT use the reverse order for grid - for easy later blending
  tour_orders <- as.vector(tour_orders)
  ## WRITE chunk size file for further blending
  chunk.sizes <- unlist(sapply(tours[tour_orders], function(t){length(labels(t))}))
  write.table(chunk.sizes, file = './chunk_sizes.csv', sep = '\r\n', row.names = F, col.names = F)
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
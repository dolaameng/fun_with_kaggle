## IDEA: right shift the region from grid_tsp.R solution, so that the twin solution
## can use the subregions that are different (right shifted) from the tsp solution.
## Some parameters are hard-coded to just prove the concept

setwd("~/workspace/x/kaggle/santa/RWithConcorde")
library(TSP)
library(doMC)
registerDoMC(cores=4)

##load the existing grided TSP solution
load('tspgridtour.RData')
## REDO GRIDING
cities <- read.csv('../data/santa_cities.csv', header = T)
## function of making equal-sized grid in the map
make.grids <- function(data.x.y, nrow, ncol, xrange=20000, yrange=20000) {
  #x.breaks <- (0:ncol) * xrange / ncol
  x.breaks <- c(0, xrange/(2*ncol) + (0:ncol) * xrange / ncol)
  y.breaks <- (0:nrow) * yrange / nrow
  x.grids <- as.integer(cut(data.x.y$x, breaks=x.breaks, right=T, include.lowest=T))
  y.grids <- as.integer(cut(data.x.y$y, breaks=y.breaks, right=T, include.lowest=T))
  data.x.y$xregion <- as.factor(x.grids)
  data.x.y$yregion <- as.factor(y.grids)
  return (data.x.y)
}

## redo city grids based on new nrow, ncol

city.grids <- make.grids(cities, nrow = nrow, ncol = ncol)
ncol <- ncol + 1 ## one more column because all the columns are halfly shifted
grid.index <- matrix(nrow = 2, ncol = nrow*ncol)
for (xgrid in 1:ncol) {
  for (ygrid in 1:nrow) {
    grid.index[,(ygrid-1)*ncol + xgrid] <- c(xgrid, ygrid)
  }
}
## NOW the number of columns are plus 1
##-- build the tour_orders - row by row (the order of grided.tours)
##--tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
##--tour_orders = as.vector(tour_orders)
## make the distance matrix entry on existing tour VERY LARGE
BIG.NUM <- Inf#100000000 # within 2k x 2k
jam_dist_matrix <- function(dist.matrix, grid.tour){
  dist.matrix <- as.matrix(dist.matrix)
  #stopifnot(nrow(dist.matrix) == length(grid.tour))
  for(i in 1:(length(grid.tour)-1)) {
    city1 <- as.character(grid.tour[i])
    city2 <- as.character(grid.tour[i+1])
    #if (city1 %in% rownames(dist.matrix) && city2 %in% rownames(dist.matrix)) {
    tryCatch({
      dist.matrix[city1, city2] <- BIG.NUM
      dist.matrix[city2, city1] <- BIG.NUM
    }, error = function(e){})
    #}
  }
  return (as.dist(dist.matrix))
}
## solve TSP for twins
twin.tours <- foreach(xy = grid.index, .packages=c('TSP'), .inorder = F) %dopar% {
  xindex <- xy[1]
  yindex <- xy[2]
  #cat(xindex, ',', yindex, '\n')
  dist.matrix <- with(city.grids, dist(city.grids[xregion==xindex & yregion==yindex, 2:3]))
  if (xindex == 1) {
    grid.tour <- grided.tours[[(yindex-1) * (ncol-1) + xindex]]
    #cat((yindex-1) * (ncol-1) + xindex, '\n')
  } else if (xindex == ncol) {
    grid.tour <- grided.tours[[(yindex-1) * (ncol-1) + xindex-1]]
    #cat((yindex-1) * (ncol-1) + xindex-1, '\n')
  } else {
    grid.tour <- c(grided.tours[[(yindex-1) * (ncol-1) + xindex-1]], grid.tour <- grided.tours[[(yindex-1) * (ncol-1) + xindex]])
    #cat((yindex-1) * (ncol-1) + xindex-1, ',', (yindex-1) * (ncol-1) + xindex, '\n')
  }
  #print('----------------------')
  dist.matrix <- jam_dist_matrix(dist.matrix, grid.tour)
  tsp <- TSP(dist.matrix)
  tour <- solve_TSP(tsp, method="nn")
  as.integer(labels(tour))
}



complete_tour <- function(grid.index, tsp.tours, city.grids) {
  tour_orders <- sapply(1:nrow, function(r){if (r%%2==1) (r-1)*ncol+seq(1,ncol) else (r-1)*ncol+seq(ncol, 1)})
  #wrong - tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
  tour_orders = as.vector(tour_orders)
  tour.labels <- tsp.tours
  
  ## WRITE chunk size file for further blending
  chunk.sizes <- unlist(sapply(tour.labels[tour_orders], function(t){length(t)}))
  write.table(chunk.sizes, file = './chunk_sizes.csv', sep = '\r\n', row.names = F, col.names = F)
  
  full.tour <- unlist(tour.labels[tour_orders])
  return (full.tour)
}

twin.tour <- complete_tour(grid.index, twin.tours, city.grids)

## save the tour to csv
write.table(twin.tour, file = './twintour.csv', sep = '\r\n', col.names = F, row.names = F)

## plotting
regioin.colors <- with(city.grids, (as.integer(xregion)-1)*10+as.integer(yregion))
with(city.grids, plot(x, y, pch='.', col=regioin.colors))
with(city.grids, lines(x[twin.tour], y[twin.tour], col=regioin.colors))
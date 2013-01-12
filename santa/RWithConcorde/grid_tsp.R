setwd("~/workspace/fun_with_kaggle/santa/RWithConcorde")

library(TSP)
library(doMC)
registerDoMC(cores=4)
## function of making equal-sized grid in the map
make.grids <- function(data.x.y, nrow, ncol, xrange=20000, yrange=20000) {
  x.breaks <- (0:ncol) * xrange / ncol
  #x.breaks <- c(0, 1000 + (0:ncol) * xrange / ncol)
  y.breaks <- (0:nrow) * yrange / nrow
  x.grids <- as.integer(cut(data.x.y$x, breaks=x.breaks, right=T, include.lowest=T))
  y.grids <- as.integer(cut(data.x.y$y, breaks=y.breaks, right=T, include.lowest=T))
  data.x.y$xregion <- as.factor(x.grids)
  data.x.y$yregion <- as.factor(y.grids)
  return (data.x.y)
}
## load city coordinates
cities <- read.csv('../data/santa_cities.csv', header = T)
## make grids
nrow <- 10
ncol <- 10
city.grids <- make.grids(cities, nrow = nrow, ncol = ncol)
## test internal tsp solutions

#nrow <- 2
#ncol <- 2
grid.index <- matrix(nrow = 2, ncol = nrow*ncol)
for (xgrid in 1:ncol) {
  for (ygrid in 1:nrow) {
    grid.index[,(ygrid-1)*ncol + xgrid] <- c(xgrid, ygrid)
  }
}
tsp.tours <- foreach(xy = grid.index, .packages=c('TSP'), .inorder = F) %dopar% {
  xindex <- xy[1]
  yindex <- xy[2]
  tsp <- with(city.grids, TSP(dist(city.grids[xregion==xindex & yregion==yindex, 2:3])))
  tour <- solve_TSP(tsp, method="linkern")
  #list(tsp = tsp, tour = tour)
  as.integer(labels(tour))
}


complete_tour <- function(grid.index, tsp.tours, city.grids) {
  tour_orders <- sapply(1:nrow, function(r){if (r%%2==1) (r-1)*ncol+seq(1,ncol) else (r-1)*ncol+seq(ncol, 1)})
  #wrong - tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
  tour_orders = as.vector(tour_orders)
  tour.labels <- tsp.tours

  full.tour <- unlist(tour.labels[tour_orders])
  return (full.tour)
}

tsp.tour <- complete_tour(grid.index, tsp.tours, city.grids)


## save the raw soltuion - tours in each grid
grided.tours <- tsp.tours
save(list = c('grided.tours', 'city.grids', 'grid.index', 'ncol', 'nrow'), file = 'tspgridtour.RData')
## save the tour to csv
write.table(tsp.tour, file = './tsptour.csv', sep = '\r\n', col.names = F, row.names = F)

## plotting
regioin.colors <- with(city.grids, (as.integer(xregion)-1)*10+as.integer(yregion))
with(city.grids, plot(x, y, pch='.', col=regioin.colors))
with(city.grids, lines(x[tsp.tour], y[tsp.tour], col=regioin.colors))


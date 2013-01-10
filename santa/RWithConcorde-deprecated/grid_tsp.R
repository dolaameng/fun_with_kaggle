setwd("~/workspace/fun_with_kaggle/santa/RWithConcorde")

library(TSP)
library(doMC)
registerDoMC(cores=4)
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
## load city coordinates
cities <- read.csv('../data/santa_cities.csv', header = T)
## make grids
nrow <- 6
ncol <- 6
city.grids <- make.grids(cities, nrow = nrow, ncol = ncol)
## test internal tsp solutions

#nrow <- 2
#ncol <- 2
grid.index <- matrix(nrow = 2, ncol = nrow*ncol)
for (xgrid in 1:ncol) {
  for (ygrid in 1:nrow) {
    grid.index[,(xgrid-1)*ncol + ygrid] <- c(xgrid, ygrid)
  }
}
tsp.tours <- foreach(xy = grid.index, .packages=c('TSP'), .inorder = F) %dopar% {
  xindex <- xy[1]
  yindex <- xy[2]
  tsp <- with(city.grids, TSP(dist(city.grids[xregion==xindex & yregion==yindex, 2:3])))
  tour <- solve_TSP(tsp, method="linkern")
  list(tsp = tsp, tour = tour)
}

## nn: 37568.60 55999.11 91206.78 69731.68 -> 254,506.2 (4 grid), 7400420 (100 grid) - super fast
## 2-opt: 227,597.9 (4 grid) - slow
## linkern: 208,393 (4 grid) - fast 6,044,707 (100 grid)
#sum(sapply(tsp.tours, function(p.t){tour_length(p.t[[1]], p.t[[2]])}))

complete_tour <- function(grid.index, tsp.tours, city.grids) {
  tsps <- sapply(tsp.tours, function(p.t){p.t[[1]]})
  tours <- sapply(tsp.tours, function(p.t){p.t[[2]]})
  inner_tours_length = sum(sapply(tsp.tours, function(p.t){tour_length(p.t[[1]], p.t[[2]])}))
  #wrong - tour_orders = sapply(1:nrow, function(r){if (r%%2==1) (r-1)*ncol+seq(1,ncol) else (r-1)*ncol+seq(ncol, 1)})
  tour_orders = sapply(1:ncol, function(c){if (c%%2==1) (c-1)*nrow+seq(1,nrow) else (c-1)*nrow+seq(nrow, 1)})
  tour_orders = as.vector(tour_orders)
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

tsp.tour <- complete_tour(grid.index, tsp.tours, city.grids)
print (tsp.tour$full.tour.length)
print (length(tsp.tour$full.tour))
## validate the distance of the full tour -- slow
##get_tour_length <- function(tour, cities) {
##  total.length <- foreach (i = 1:(length(tour)-1), .combine='sum', .inorder=F) %dopar% {
##    dist(cities[c(tour[i], tour[i+1]), 2:3])[[1]]
##  }
##  return (total.length)
##}
##print(get_tour_length(tsp.tour$full.tour, cities))

## save the raw soltuion - tours in each grid
grided.tours <- sapply(tsp.tours, function(t){labels(t$tour)})
save(list = c('grided.tours', 'city.grids', 'grid.index', 'ncol', 'nrow'), file = 'tspgridtour.RData')
## save the tour to csv
write.table(tsp.tour$full.tour, file = './tsptour.csv', sep = '\r\n', col.names = F, row.names = F)

## plotting
regioin.colors <- with(city.grids, (as.integer(xregion)-1)*10+as.integer(yregion))
with(city.grids, plot(x, y, pch='.', col=regioin.colors))
with(city.grids, lines(x[tsp.tour$full.tour], y[tsp.tour$full.tour], col=regioin.colors))
title(paste('tour with distance ', tsp.tour$full.tour.length))

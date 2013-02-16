setwd("~/workspace/fun_with_kaggle/event-recommendation/R")
library(bigmemory)
library(biganalytics)
library(foreach)
library(doMC)
registerDoMC(cores=4)

from.start = F

## NO NEED TO GENERATE IT EVERY TIME
## generate event_ids.csv and event_words.csv from events.csv

events_path <- '../data/events.csv'
#events_path <- '../data/train_test_events.csv'


if(from.start) {
  print('extract information from events.csv to generate event_ids.csv and event_words.csv')
  system(paste('cat ', events_path, ' | cut -d , -f 10- > ../data/event_words.csv', sep=''))
  system(paste('cat ', events_path, ' | cut -d , -f 1 > ../data/event_ids.csv', sep=''))
}

## load big matrix
event.words <- read.big.matrix('../data/event_words.csv', sep=',', header=T, type='integer')

## kmeans clustering - about 15 mins
k = 32 ## SIMPLY BECAUSE RANDOMFOREST IN R DOEST ALLOW MORE THAN 32 LEVELS
event.clusters <- bigkmeans(event.words, centers=k, nstart=4, iter.max = 40)


## write results
events <- read.csv('../data/event_ids.csv', header = T)
events <- transform(events, cluster=event.clusters$cluster)
write.csv(events, file=paste('../data/event_clusters',k,'.csv',sep=''), row.names = F)
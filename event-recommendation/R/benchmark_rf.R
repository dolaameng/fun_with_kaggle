
setwd("~/workspace/fun_with_kaggle/event-recommendation/R")
library(randomForest)
library(foreach)
#library(Hmisc)
library(doMC)
library(rminer)
registerDoMC(cores = 4)

## common setting for train and test

rmse <- function(obs, pred) {return(sqrt(mean((obs-pred)^2)))}
#{imputation (knn-3), na.roughfix, imputate}
imputation.method <- function(data){return (imputation(imethod='hotdeck',data,Value=3))} 

## load test data
test.set <- read.csv('../data/test_classifier.csv', header=T)
## load train data
train.set <- read.csv('../data/train_classifier.csv', header=T)
locale.levels = union(levels(train.set$user_local), levels(test.set$user_local))
train.set <- transform(train.set, user_locale=factor(user_locale, levels=locale.levels))
test.set <- transform(test.set, user_locale=factor(user_locale, levels=locale.levels))
## handle missing values - mainly in gender and age
train.set <- imputation.method(train.set)
test.set <- imputation.method(test.set)
features <- -grep("^*(user|event)$", names(train.set)) #exclude non-use features

## as a regression, because interest_rank now is int 
train.set_validation.index <- sample(nrow(train.set), nrow(train.set)/5, replace = F)
train.set_train <- train.set[-train.set_validation.index, features]
train.set_validation <- train.set[train.set_validation.index, features]

## build randomForest model in parallel
rf.model <- foreach(ntree=rep(200,4), .combine=combine, .packages="randomForest", .inorder=F) %dopar% {
  randomForest(interest_rank~., data=train.set_train, 
               ntree=ntree, importance=T, na.action=na.roughfix, replace = F)
}
## validate the built model
print (rmse(train.set_validation$interest_rank, 
           predict(rf.model, newdata=train.set_validation, type='response')))
## predict on test data
test.prediction <- predict(rf.model, newdata=test.set, type='response', na.action=na.roughfix)
test.set <- transform(test.set, interest_rank=test.prediction)
## write test to csv
write.csv(test.set, '../data/test_predictions.csv', row.names=F)

setwd("~/workspace/fun_with_kaggle/event-recommendation/R")
library(randomForest)
library(foreach)
#library(Hmisc)
library(doMC)
library(rminer)
registerDoMC(cores = 4)

## common setting for train and test

rmse <- function(obs, pred) {return(sqrt(mean((as.numeric(obs)-as.numeric(pred))^2)))}
#{imputation (knn-3), na.roughfix, imputate}
imputation.method <- function(data){return (imputation(imethod='hotdeck',data,Value=3))} 

## load test data
#test.set <- read.csv('../data/test_classifier.csv', header=T)
test.set <- read.csv('../data/test_private_classifier.csv', header=T)
## load train data
train.set <- read.csv('../data/train_classifier.csv', header=T)
## factorize
locale.levels <- union(levels(train.set$user_local), levels(test.set$user_local))
train.set$topic <- as.factor(train.set$topic)
test.set$topic <- as.factor(test.set$topic)
topic.levels <- union(levels(train.set$topic), levels(test.set$topic))
community.levels <- union(levels(train.set$user_community), levels(test.set$user_community))
train.set <- transform(train.set, user_locale=factor(user_locale, levels=locale.levels),
                                  topic=factor(topic, levels=topic.levels, ordered=F),
                       user_community=factor(user_community, levels=community.levels, ordered=F))
test.set <- transform(test.set, user_locale=factor(user_locale, levels=locale.levels),
                                topic=factor(topic, levels=topic.levels, ordered=F),
                        user_community=factor(user_community, levels=community.levels, ordered=F))
## handle missing values - mainly in gender and age
## invalid ages >= 50
train.set$user_age <- ifelse(train.set$user_age <= 50,train.set$user_age, NA)
test.set$user_age <- ifelse(test.set$user_age <= 50,test.set$user_age, NA)
train.set <- imputation.method(train.set)
test.set <- imputation.method(test.set)
features <- -grep("^*(user|event)$", names(train.set)) #exclude non-use features

## as a regression, because interest_rank now is int 
train.set_validation.index <- sample(nrow(train.set), nrow(train.set)/5, replace = F)
train.set_train <- train.set[-train.set_validation.index, features]
train.set_validation <- train.set[train.set_validation.index, features]

## build randomForest model in parallel
## use whole set to train
rf.model <- foreach(ntree=rep(400,8), .combine=combine, .packages="randomForest", .inorder=F) %dopar% {
  randomForest(interest_rank~., data=train.set[,features], 
               ntree=ntree, importance=T, na.action=na.roughfix)
}
## validate the built model
print (rmse(train.set_validation$interest_rank, 
           predict(rf.model, newdata=train.set_validation, type='response')))
## predict on test data
test.prediction <- predict(rf.model, newdata=test.set, type='response')
test.set <- transform(test.set, interest_rank=test.prediction)
## write test to csv
write.csv(test.set, '../data/test_predictions.csv', row.names=F)
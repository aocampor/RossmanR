library(lubridate)
library(Matrix)
library(xgboost)
library(reshape2)
library(ggplot2)

#reading csv files 
train <- read.csv(file="/home/aocampor/workspace/Rossman/Data/train.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))
test <- read.csv(file="/home/aocampor/workspace/Rossman/Data/test.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))
store <- read.csv(file="/home/aocampor/workspace/Rossman/Data/store.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))

#adding Data variables
train$Date  <- strptime(train$Date,format = "%Y-%m-%d", tz="GMT")
train$year <- year(as.Date(train$Date))
train$month <- month(as.Date(train$Date))
train$week <- week(as.Date(train$Date))
train$day <- day(as.Date(train$Date))

test$Date  <- strptime(test$Date,format = "%Y-%m-%d", tz="GMT")
test$year <- year(as.Date(test$Date))
test$month <- month(as.Date(test$Date))
test$week <- week(as.Date(test$Date))
test$day <- day(as.Date(test$Date))

train$Date <- NULL
test$Date <- NULL

#merging datasets
train <- merge(train, store, by="Store") 
test <- merge(test, store, by="Store") 

train$Customers <- NULL
train <- subset(train, train$Open == 1 )
testClosed <- subset(test, test$Open == 0 )
testClosed$Prediction <- 0
testNA <- subset(test , is.na(test$Open) )
testNA$Prediction <- 0 
testOpen <- subset(test, test$Open == 1 )

columns <- c("StateHoliday", "StoreType", "Assortment", "PromoInterval")
for(item in columns){
  train[item] <- as.numeric(train[[item]])
}
for(item in columns){
  testOpen[item] <- as.numeric(testOpen[[item]])
}

#columns <- c("StateHoliday", "Promo2", "PromoInterval", "SchoolHoliday")#, 
             #"Assorment", "Promo", "StoreType", "year", "Promo2SinceYear", 
             #"month", "Promo2SinceWeek", "CompetitionOpenSinceMonth",
             #"DayOfWeek", "week", "CompetitionOpenSinceYear", "day", 
             #"CompetitionDistance", "Store")

#for(item in columns){
#  train[item] <- NULL
#}
#for(item in columns){
#  testOpen[item] <- NULL
#}

#train <- train[train$Sales < 17000,]
sales <- train$Sales
train$Sales <- NULL

#hist(train$Sales)
#100*nrow(train[train$Sales > 17000,])/nrow(train)

id <- testOpen$Id
testOpen$Id <- NULL
sales.log <- log(sales + 1)

## treating cost as log transfromation is working good on this data set
trainMatrix <- as.matrix(train)
testMatrix <- as.matrix(testOpen)

tr.x <- xgb.DMatrix(trainMatrix, label = sales.log, missing=NA)
te.x <- xgb.DMatrix(testMatrix, missing=NA)

## parameter selection
#par  <-  list(booster = "gbtree", objective = "reg:linear", 
#              min_child_weight = 0.02, eta = 0.12, gamma = 0.002,
#              max_depth = 21)
par  <-  list(booster = "gbtree", objective = "reg:linear", 
              min_child_weight = 0.2, eta = 0.5, gamma = 0.5,
              #subsample = 0.0014, colsample_bytree = 0.12,
              max_depth = 21, max_delta_step = 0.1,
              verbose = 1, scale_pos_weight = 1, eval_metric = "rmse")

##selecting number of Rounds
n_rounds= 200 #10000  #nrow(train)

######################################
###In case a new training is needed
ptm <- proc.time()
x.mod.t  <- xgb.train(params = par, data = tr.x , nrounds = n_rounds)
#x.mod.t  <- xgboost(params = par, data = tr.x , nrounds = n_rounds)
#cvxgb <- xgb.cv(params = par, data = tr.x , nrounds = n_rounds, nfold = 4)
#str(cvxgb)
#cvxgb$rank <- c(1:nrow(cvxgb))
#cvxgb1 <- cvxgb[cvxgb$rank > 6000]
#cvxgb1 <- cvxgb1[cvxgb1$rank < 6900]
#plot(cvxgb1$rank, cvxgb1$test.rmse.mean)# , ylim=c(0.4273,0.4274))
#points(cvxgb1$rank, cvxgb1$test.rmse.mean , col = "red")
#names <- names(train)
#xgb.importance(names, model = x.mod.t)
#points(cvxgb1$rank, cvxgb1$train.rmse.mean, col=2)
#xgb.save(x.mod.t, '/home/aocampor/workspace/Rossman/src/trained1483.model')
proc.time() - ptm 
pred <- predict(x.mod.t, te.x)

###########################################
####In case we are loading something pretrained
x.mod.t  <- xgb.load('/home/aocampor/workspace/Rossman/src/train_nrounds200_05etagamma.model')
pred <- predict(x.mod.t, te.x)

for(i in 1:10){
  x.mod.t  <- xgb.train(par,tr.x,n_rounds)
  pred  <- cbind(pred,predict(x.mod.t,te.x))
}
pred.sub  <- exp(rowMeans(pred))-1
#pred.sub  <- exp(pred)-1
preds <- c( pred.sub, testClosed$Prediction, testNA$Prediction)
ids <- c(id, testClosed$Id, testNA$Id)

## generating data frame for submission
sub.file = data.frame(Id = ids, Sales = preds)
#sub.file
#sub.file = aggregate( data.frame( Sales = sub.file$Sales), by = list(Id = sub.file$Id), mean)
write.csv(sub.file, "/home/aocampor/workspace/Rossman/benchmark_allVariables_nrounds200_05etagamma.csv", row.names = FALSE, quote = FALSE)
#hist(sub.file$Sales)
#hist(pred.sub)

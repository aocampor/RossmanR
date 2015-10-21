#library("e1071")
library(lubridate)
library(Matrix)
library(xgboost)
library(reshape2)
library(ggplot2)
library(kernlab)

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

train <- train[train$Sales < 17000,]
sales <- train$Sales
train$Sales <- NULL

#hist(train$Sales)
#100*nrow(train[train$Sales > 17000,])/nrow(train)

id <- testOpen$Id
testOpen$Id <- NULL
sales.log <- log(sales + 1)

trainMatrix <- as.matrix(train)
testMatrix <- as.matrix(testOpen)
str(sales)

ptm <- proc.time()
svp <- ksvm(trainMatrix, sales, type="C-svc",kernel='rbfdot',C=100,scaled=c())
proc.time() - ptm

pred = predict(svp,testMatrix)

preds <- c( pred, testClosed$Prediction, testNA$Prediction)
ids <- c(id, testClosed$Id, testNA$Id)

## generating data frame for submission
sub.file = data.frame(Id = ids, Sales = preds)

#sub.file = aggregate( data.frame( Sales = sub.file$Sales), by = list(Id = sub.file$Id), mean)
write.csv(sub.file, "/home/aocampor/workspace/Rossman/src/svn_salesless17000.csv", row.names = FALSE, quote = FALSE)
#svmfit <- svm(sales.log, data = train, kernel = "linear", cost = 10, scale = FALSE)
#plot(svmfit, train)

## treating cost as log transfromation is working good on this data set
#trainMatrix <- as.matrix(train)
#testMatrix <- as.matrix(testOpen)

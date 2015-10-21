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
str(sales)

ptm <- proc.time()
svp <- ksvm(trainMatrix, sales, type="C-svc",kernel='rbfdot',C=100,scaled=c())
proc.time() - ptm

#svmfit <- svm(sales.log, data = train, kernel = "linear", cost = 10, scale = FALSE)
#plot(svmfit, train)

## treating cost as log transfromation is working good on this data set
#trainMatrix <- as.matrix(train)
#testMatrix <- as.matrix(testOpen)


###############################

n <- 150 # number of data points
p <- 2

sigma <- 1 # variance of the distribution
meanpos <- 0 # centre of the distribution of positive examples
meanneg <- 3 # centre of the distribution of negative examples
npos <- round(n/2) # number of positive examples
nneg <- n-npos # number of negative examples
# Generate the positive and negative examples
xpos <- matrix(rnorm(npos*p,mean=meanpos,sd=sigma),npos,p)
xneg <- matrix(rnorm(nneg*p,mean=meanneg,sd=sigma),npos,p)
x <- rbind(xpos,xneg)

# Generate the labels
y <- matrix(c(rep(1,npos),rep(-1,nneg)))
# Visualize the data
plot(x,col=ifelse(y>0,1,2))
legend("topleft",c('Positive','Negative'),col=seq(2),pch=1,text.col=seq(2))
## Prepare a training and a test set ##
ntrain <- round(n*0.8) # number of training examples
tindex <- sample(n,ntrain) # indices of training samples
xtrain <- x[tindex,]
xtest <- x[-tindex,]
ytrain <- y[tindex]
ytest <- y[-tindex]
istrain=rep(0,n)
istrain[tindex]=1
# Visualize
plot(x,col=ifelse(y>0,1,2),pch=ifelse(istrain==1,1,2))
legend("topleft",c('Positive Train','Positive Test','Negative Train','Negative Test'),
       col=c(1,1,2,2),pch=c(1,2,1,2),text.col=c(1,1,2,2))
# load the kernlab package
library(kernlab)
# train the SVM
ptm <- proc.time()
svp <- ksvm(xtrain,ytrain,type="C-svc",kernel='vanilladot',C=100,scaled=c())
proc.time() - ptm
attributes(svp)
plot(svp,data=xtrain)
# Predict labels on test
ypred = predict(svp,xtest)
table(ytest,ypred)
# Compute accuracy
sum(ypred==ytest)/length(ytest)
# Compute at the prediction scores
ypredscore = predict(svp,xtest,type="decision")
# Check that the predicted labels are the signs of the scores
table(ypredscore > 0,ypred)
# Package to compute ROC curve, precision-recall etc...
library(ROCR)
pred <- prediction(ypredscore,ytest)
# Plot ROC curve
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf)
# Plot precision/recall curve
perf <- performance(pred, measure = "prec", x.measure = "rec")
plot(perf)
# Plot accuracy as function of threshold
perf <- performance(pred, measure = "acc")
plot(perf)

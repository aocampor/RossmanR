library(lubridate)
library(Matrix)
library(xgboost)
library(reshape2)
library(ggplot2)

train <- read.csv(file="/home/aocampor/workspace/Rossman/Data/train.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))
store <- read.csv(file="/home/aocampor/workspace/Rossman/Data/store.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))
test <- read.csv(file="/home/aocampor/workspace/Rossman/Data/test.csv",head=TRUE,sep=",", na.strings=c("NA", "NULL"))

names(train)
names(test)
names(store)

#adding Data variables
train$Date  <- strptime(train$Date,format = "%Y-%m-%d", tz="GMT")
train$year <- year(as.Date(train$Date))
train$month <- month(as.Date(train$Date))
train$week <- week(as.Date(train$Date))
train$day <- day(as.Date(train$Date))

train$Date <- NULL

#days <- c(1,2,3,4,5,6,7)
#means <- NULL
#for(it in days){
#  firstday <- train[train$DayOfWeek == it, ]
#  means <- c(means, mean(firstday$Sales))
#}
#means.df <-  data.frame(DayOfWeek = days, MeanSales = means)
#means.df$MeanSales <- log(means.df$MeanSales)
#str(means.df)

#merging datasets
train <- merge(train, store, by="Store") 
#train <- merge(train, means.df, by="DayOfWeek")
#str(train)
#hist(train$MeanSales)

set.seed(10)
trainTemp <- split(train, sample(rep(1:2, 508604)))

training <- trainTemp$`1`
testing <- trainTemp$`2`

columns <- c("StateHoliday", "StoreType", "Assortment", "PromoInterval")
for(item in columns){
  training[item] <- as.numeric(training[[item]])
}
for(item in columns){
  testing[item] <- as.numeric(testing[[item]])
}

training <- training[ training$Open == 1 , ]
training$Customers <- NULL
testing$Customers <- NULL

#cor(training, use="pairwise.complete.obs")
#names_training <- names(training)
#for(na in names_training){
#  corre <- cor(training$Sales, training[na], use="pairwise.complete.obs")
#  if(is.na(corre))
#    next
#  if(abs(corre) < 0.01){
#    training[na] <- NULL
#    testing[na] <- NULL
#  }
#}
#cor(training, use="pairwise.complete.obs")
#plot(training$Sales, training$CompetitionDistance)
#nrow(testing)

#hist(training$Sales)
#100*nrow(training[training$Sales > 17000,])/nrow(training)
#training <- training[training$Sales < 17000, ]
sales <- training$Sales
training$Sales <- NULL
#plot(sales,training$MeanSales,xlab="Sales",ylab="Mean Sales",col=abs(sales-training$MeanSales))

testingClose <- testing[ testing$Open == 0 , ]
testingClose$Prediction <- 0
testingOpen <- testing[ testing$Open == 1 , ]
training$Open <- NULL
testingOpen$Open <- NULL

ctr <- testingOpen$Sales
testingOpen$Sales <- NULL

sales.log <- log(sales + 1)
ctr.log <- log(ctr + 1)
#nrow(testingOpen)

#training$MeanSales <- NULL
#testingOpen$MeanSales <- NULL

names <- names(training)
for(nam in names){
  plot(training$Sales, training[,nam], xlab = "Sales", ylab = nam)
}

trainingM <- as.matrix(training)
testingM <- as.matrix(testingOpen)

training.x <- xgb.DMatrix(trainingM, label = sales.log , missing = NA)
testing.x <- xgb.DMatrix(testingM, missing = NA)

##selecting number of Rounds
errors <- NULL
xaxis <- NULL
step <- 4000
i <- 1
#for ( i in 1:1){/home/aocampor/workspace/Rossman/src/
  par  <-  list(booster = "gbtree", objective = "reg:linear", 
                min_child_weight = 0.02, eta = 0.12, gamma = 0.002, 
                subsample = 0.0014, colsample_bytree = 1, max_depth = 21, 
                max_delta_step = 0.1, verbose = 1, scale_pos_weight = 1, eval_metric = "rmse")
  #x.mod.t  <- xgb.train(params = par, data = training.x , nrounds = i*step)
  x.mod.t  <- xgboost(params = par, data = training.x , nrounds = 4000)
  cvxgb <- xgb.cv(params = par, data = training.x, nrounds = step, nfold = 3)
  pred <- predict(x.mod.t, testing.x)
  xaxis <- c(xaxis, i*step)
  errors <- c(errors,sqrt( mean( (pred - ctr.log)^2 ) ) )
#}
cvxgb1
plot(cvxgb1)
cvxgb$index <- order 
cvxgb1 <- cvxgb[ cvxgb$index > 1250, ]
cvxgb1 <- cvxgb1[ cvxgb1$index < 1400, ]

names <- names(training)
xgb.importance(names, model = x.mod.t)
names(test)

order <- c(1:nrow(cvxgb))
plot(cvxgb1$index, cvxgb1$test.rmse.mean)
lines(cvxgb1$index, cvxgb1$test.rmse.mean)
plot(cvxgb$test.rmse.mean, cvxgb$test.rmse.std)

cvxgb$Order <- order

cvxgb1 <- cvxgb[cvxgb$test.rmse.std < 0.002 , ]
cvxgb2 <- cvxgb1[cvxgb1$test.rmse.mean < 1, ]
nrow(cvxgb2)
cvxgb2$Ranking <-sqrt( cvxgb2$train.rmse.mean * cvxgb2$train.rmse.mean + cvxgb2$train.rmse.std * cvxgb2$train.rmse.std  )  
cvxgb2

plot(xaxis, errors, xlab="subsample", ylab="errors")
heat <- topo.colors(40000, alpha = 1)
color <- NULL
for(i in 1:length(ctr.log)){
  color <- c(color, heat[[abs(exp(ctr.log[[i]])- exp(pred[[i]]))]])
}
plot( exp(ctr.log) - 1, exp(pred) - 1, ylab="Prediction", xlab = "Control Value Summer",col = color,  pch=19)
abline(0, 1)
cor(ctr.log, pred)

#hist(training$MeanSales)

testingOpen$Control <- ctr.log
testingOpen$Prediction <- pred
testingOpen$Sales <- ctr

#temp <- testingOpen[testingOpen$Control == 0 , ]
#temp1 <- testingOpen[testingOpen$Control > 0 , ] 
#str(temp)
#hist(temp$DayOfWeek)
#hist(temp1$DayOfWeek)
#nrow(temp)/nrow(temp1)*100

#temp <- testing[ testing$Control < 1,  ]
#temp1 <- testing[ testing$Control > 1 & testing$Prediction > 7, ]
#temp2 <- testing[ testing$Control > 1 & testing$Prediction <= 7 , ]

#saturday <- testing[ testing$DayOfWeek == 7 , ]
#weeko <- testing[ testing$DayOfWeek < 7, ]

#temp3 <- testing[ testing$Open == 1 & testing$DayOfWeek == 7,]
#temp4 <- testing[ testing$Open == 0 & testing$DayOfWeek == 7,]
#temp5 <- testing[ testing$Open == 1 & testing$DayOfWeek < 7 , ]
#temp6 <- testing[ testing$Open == 0 & testing$DayOfWeek < 7 , ]
#temp7 <- testing[ testing$Open == 0 , ]
#hist(temp7$Prediction )
#str( temp5$Sales )

#names <- names(temp1)
#for(it in names){
#  plot( testingOpen[[it]], pred, ylab="Prediction", col = testingOpen[[it]] - pred,  xlab = it, pch=19)
#  plot( testingOpen[[it]], ctr.log, ylab="Control", col = testingOpen[[it]] - ctr.log, xlab = it, pch=19)
  
  #hist( saturday[[it]], xlab = it )
  #hist( weeko[[it]], xlab = it)
#}

#for(it in names){
#  hist(temp[[it]] , xlab = it)
#  hist(temp1[[it]] , xlab = it)
#  hist(temp2[[it]] , xlab = it)
#}
testingOpen$Color <- abs(ctr.log - pred)*10
testingOpen$Color <- 1 - abs(pred - ctr.log)/ctr.log
plot(testingOpen$Color, testingOpen$StateHoliday)
hist(1 - abs(pred - ctr.log)/ctr.log, breaks=100)


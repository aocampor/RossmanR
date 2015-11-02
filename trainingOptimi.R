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

#merging datasets
train <- merge(train, store, by="Store") 

#set.seed(10)
trainTemp <- split(train, sample(rep(1:2, 0.8*nrow(train))))
training <- trainTemp$`1`
testing <- trainTemp$`2`

columns <- c("StateHoliday", "StoreType", "Assortment", "PromoInterval")
for(item in columns){
  training[item] <- as.numeric(training[[item]])
}

for(item in columns){
  testing[item] <- as.numeric(testing[[item]])
}

#names(training)
columns <- c("StateHoliday", "Promo2", "PromoInterval", "SchoolHoliday", 
             "Assorment", "Promo", "StoreType", "year", "Promo2SinceYear", 
             "month", "Promo2SinceWeek", "CompetitionOpenSinceMonth",
             "DayOfWeek", "week", "CompetitionOpenSinceYear")

for(item in columns){
  training[item] <- NULL
  testing[item] <- NULL
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
hist(sales.log, breaks = 100)
#training$MeanSales <- NULL
#testingOpen$MeanSales <- NULL

names <- names(training)
#length(sales)
#nrow(training)
for(nam in names){
  plot(sales, training[,nam], xlab = "Sales", ylab = nam)
}
#plot(training)

trainingM <- as.matrix(training)
testingM <- as.matrix(testingOpen)

training.x <- xgb.DMatrix(trainingM, label = sales.log , missing = NA)
testing.x <- xgb.DMatrix(testingM, missing = NA)

##selecting number of Rounds
errors <- NULL
xaxis <- NULL
step <- 1
rounds = 400
i <- 1
#for (i in 1:10){
  par  <-  list(booster = "gbtree", objective = "reg:linear", 
                min_child_weight = step, eta = 0.5, gamma = 0.5, 
                #subsample = 0.0014, colsample_bytree = 1, 
                max_depth = 21, 
                max_delta_step = 0.1, 
                verbose = 1, scale_pos_weight = 1, eval_metric = "rmse")
  x.mod.t  <- xgb.train(params = par, data = training.x , nrounds = rounds)
  #xgb.save(x.mod.t, '/home/aocampor/workspace/Rossman/src/train_nrounds4000_v.1Par_using80per_4variables.model')
  #xgb.importance(names, model = x.mod.t)
  pred <- predict(x.mod.t, testing.x)
  #names(training)
  cvxgb <- xgb.cv(params = par, data = training.x, nrounds = rounds, nfold = 3)
  cvxgb$index <- c(1:nrow(cvxgb)) 
  #cvxgb1 <- cvxgb[ cvxgb$index > 800, ]
  #cvxgb1 <- cvxgb1[ cvxgb1$index < 4000, ]
  plot(cvxgb$index, log(cvxgb$train.rmse.mean), main = "Log rmse mean")
  points(cvxgb$index, log(cvxgb$test.rmse.mean), col="red")
  plot(log(cvxgb$train.rmse.mean), log(cvxgb$train.rmse.std), main="std vs mean")

  heat <- topo.colors(40000, alpha = 1)
  comparisons <- data.frame(Control=exp(ctr.log)-1, Prediction=exp(pred)-1)
  comparisons$diff <- comparisons$Control - comparisons$Prediction
  #errors <- c(errors, mean(comparisons$diff))
  #xaxis <- c(xaxis, i)
  
  color <- heat[abs(comparisons$Control - comparisons$Prediction)]
  plot( exp(ctr.log) - 1, exp(pred) - 1, ylab="Prediction", xlab = "Control",col = color,  pch=19)
  #plot( exp(ctr.log) - 1, exp(pred) - 1, ylab="Prediction", xlab = "Control",  pch=19)
  abline(0, 1, col = "red")

  qqplot(comparisons$Control, comparisons$Prediction)
  abline(0,1,col="red")
  #cor(ctr.log, pred)

plot(xaxis, errors)
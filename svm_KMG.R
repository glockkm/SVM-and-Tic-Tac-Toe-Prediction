#Kimberly Glock
#11/12/2019
#MSDS 5213 Lab 3 SVM

library(e1071)
library(caret)
library(ROCR)
library(readr)
library(dplyr)
library(caTools)

#download the file and save it. read into r studio
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data", "tac.data", method="curl")
data = read.table("tac.data", sep=",") 
View(data)
data.f = as.data.frame(data)
View(data.f)
 #or
tac = read.csv(file="tic-tac.csv", header=TRUE, sep=",")
View(tac)

anyNA(data.f)

#clean data (make integers) except for the class column V10
data.f$V1 = as.integer(data.f$V1) 
data.f$V2 = as.integer(data.f$V2) 
data.f$V3 = as.integer(data.f$V3) 
data.f$V4 = as.integer(data.f$V4) 
data.f$V5 = as.integer(data.f$V5) 
data.f$V6 = as.integer(data.f$V6) 
data.f$V7 = as.integer(data.f$V7) 
data.f$V8 = as.integer(data.f$V8) 
data.f$V9 = as.integer(data.f$V9) 
View(data.f)
class(data.f$V9)


#split data.f data into test/train
set.seed(101)
split = sample.split(data.f$V10, SplitRatio= 0.8)
train = subset(data.f, split==TRUE)
test = subset(data.f, split==FALSE, select=-V10)

y = train$V10 #target variable
x = subset(train, select=-V10) #without target variable

poly_mod = train(x, y, method="svmPoly", allowParallel = FALSE, tuneLength=5,
                 trControl=trainControl(method="repeatedcv", 
                 number=10, repeats=10))
poly_mod #to see performance
#BEST MODEL
#accuracy = 0.9924298
#kappa = 0.98320279 

linea_mod = train(x, y, method="svmLinear", tuneLength=5,
                  trControl=trainControl(method="repeatedcv", 
                                         number=10, repeats=10))
linea_mod
#accuracy = 0.6532157
#kappa = 0


radial_mod = train(x, y, method="svmRadial", allowParallel = FALSE, tuneLength=5,
                   trControl=trainControl(method="repeatedcv", 
                                          number=10, repeats=10))
radial_mod
#accuracy = 0.9513526
#kappa = 0.8886514


pred = predict(poly_mod, test)
y2 = subset(data.f, split==FALSE)$V10
confusionMatrix(pred, y2)
#accuracy = 0.9895
#kappa = 0.9767
#p-value = <2e-16 
#Sensitivity = 0.9697          
#Specificity = 1.0000 

#bootstrap 100 samples and calculate the 95% CI and AUC
n = 100
accuracy = rep(0,n)
auc = rep(0,n)
for (i in 1:n) {
  set.seed(i+100)
  new_index = sample(c(1:length(data.f$V10)), length(data.f$V10), replace=TRUE)
  new_sample = data.f[new_index,]
  split = sample.split(new_sample$V10, SplitRatio = 0.8)
  train = subset(new_sample, split==TRUE)
  test.x = subset(subset(new_sample, split==FALSE), select=-V10)
  test.y = subset(new_sample, split==FALSE)$V10
  
  svm_poly = train(V10~ ., data=train, method="svmPoly",
                   tuneLength = 1,
                   trControl= trainControl(method="repeatedcv",
                                                           repeats = 5, 
                                                           classProbs=TRUE))

#accuracy
pred = predict(svm_poly, test.x)
c = confusionMatrix(pred, test.y)
accuracy[i] = c[3]$overall[1]

#auc
pred = predict(svm_poly, test.x, type="prob")[,2]
out = prediction(pred, test.y)
auc[i] = performance(out, measure="auc")@y.values[[1]]
}

#95% ci for accuracy
accuracy
accuracy.mean = mean(accuracy)
accuracy.mean
#accuracy mean =  0.6533581
accuracy.me = qnorm(0.975) * sd(accuracy)/sqrt(length(accuracy)) #Divide your accuracy standard deviation by the square root of your accuracy size.
accuracy.me
accuracy.lci = accuracy.mean - accuracy.me #lower end of ci
accuracy.lci
accuracy.uci = accuracy.mean + accuracy.me #upper end of ci
accuracy.uci
#lower ci = 0.6500296
#upper ci = 0.6566867

#95% ci for auc
auc.mean = mean(auc)
auc.me = qnorm(0.975) * sd(auc)/sqrt(length(auc))
auc.lci = auc.mean - auc.me
auc.uci = auc.mean + auc.me
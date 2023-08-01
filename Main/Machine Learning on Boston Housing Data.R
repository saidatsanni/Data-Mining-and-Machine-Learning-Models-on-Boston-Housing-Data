
###########################################Analysis of Boston Housing Data Set using Machine Learning Techniques ##############################################
###Methods
#1. Linear Regression Analysis
#2. Variable Selection (Best subset, forward selection, backward selection, Stepwise, and Lasso variable selection methods)
#3. Cross Validation
#4. Regression Trees
#5. Bagging
#6. Random Forest
#7. Boosted Regression Trees
#8. Generalized Additive Model
#9. Neural Networks
##  Exploratory Data Analysis, Residual Diagnostics, In-sample Prediction, Out-of-sample Prediction, Predictive Performance, and Model Comparison are also included.


##Load the required packages

library(MASS)
library(Hmisc)
library(dplyr)
library(corrr)
library(tidyverse)
library(corrplot)
library(leaps)
library(glmnet)
library (boot)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(mgcv)
library(neuralnet)

#load the data
data(Boston)

##Split the data into 70% Training and 30% Testing data set
sample_index <- sample(nrow(Boston),nrow(Boston)*0.70)
Boston_train <- Boston[sample_index,]
Boston_test <- Boston[-sample_index,]


##########################################################################Exploratory Data Analysis
#Summary Statistics 
dim(Boston_train)
str(Boston_train)
summary(Boston_train)
apply(Boston_train[,1:14], 2, sd) #sd

#Correlation Analysis
corr_matrix<-cor(Boston_train)
corrplot(corr_matrix, type="upper")
Boston_train %>% correlate() %>% focus(medv)
cor(Boston_train$rad, Boston_train$tax)
cor(Boston_train$nox, Boston_train$age)
cor(Boston_train$dis, Boston_train$age)


#Outlier Analysis
Boston_train %>%
  gather(key = "var", value = "value") %>%
  ggplot(aes(x = '',y = value)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ var, scales = "free") +
  theme_bw()




############################################################### 1: Linear Regression Modeling
#Full Model
model_full <- lm(medv ~ ., data = Boston_train)
model_full.sum <- summary(model_full)
model_full.sum
(model_full.sum$sigma)^2 #MSE
AIC(model_full) #AIC value
BIC(model_full) #BIC value


#in-sample performance of the full model
pred_train_full <- predict(model_full, newdata = Boston_train)
mean((pred_train_full-Boston_train$medv)^2)  #MSE


#out-of sample performance of the full model
pred_test_full <- predict(model_full, newdata = Boston_test)
mean((pred_test_full-Boston_test$medv)^2) #MSPE



############################################################## 2: Variable Selection Methods
##############################Method 1: Best subset

model_subset <- regsubsets(medv~., data=Boston_train, nbest=2,nvmax = 14)
summary(model_subset)
plot(model_subset)

#Best model
model_subset.coef <- lm(medv~crim+zn+nox+rm+dis+rad+tax+ptratio+black+lstat, data=Boston_train)
summary(model_subset.coef)

#MSE, AIC, and BIC
summary_boston_subset <- summary(model_subset.coef)
(summary_boston_subset$sigma)^2 #MSE
AIC(model_subset.coef) #AIC value
BIC(model_subset.coef) #BIC value


###############################Method 2: Forward Variable Selection
model_null=lm(medv~1, data=Boston_train) 
fullmodel=lm(medv~., data=Boston_train)
model.step_forward = step(model_null, scope=list(lower=model_null, upper=fullmodel), direction="forward")


################################Method 3: Backward Selection
model.step_backward = step(fullmodel,direction="backward")


################################Method 4:Stepwise selection
model.step_both = step(model_null, scope=list(lower=model_null, upper=fullmodel), direction='both')
summary_step_both <- summary(model.step_both)
(summary_step_both$sigma)^2
AIC(model.step_both) 
BIC(model.step_both) 



################################Method 5:LASSO Variable Selection
Boston_lasso = glmnet(x = as.matrix(Boston_train[, -c(which(colnames(Boston_train)=='medv'))])
                      , y = Boston_train$medv, alpha = 1)
plot(Boston_lasso)

#Perform cross-validation to determine the shrinkage parameter. (set a seed)
x = as.matrix(Boston_train[ , 1:13])
y = Boston_train$medv
Boston_lasso.cv = cv.glmnet(x, y, alpha = 1)
plot(Boston_lasso.cv)
#Get the coefficient with optimal  lambda
coef(Boston_lasso, s=Boston_lasso.cv$lambda.min)


#In-sample prediction using variables selected by the lasso variable selection method
bestlam =Boston_lasso.cv$lambda.min 
lasso.pred_insample=predict(Boston_lasso,newx=as.matrix(   Boston_train[, -c(which(colnames(Boston_train)=='medv'))] ,s=bestlam ,type = "response"))
mean((lasso.pred_insample-Boston_train$medv)^2) #MSE


# out-of-sample prediction using variables selected by the lasso variable selection method
lasso.pred_outsample=predict(Boston_lasso,newx=as.matrix(Boston_test[, -c(which(colnames(Boston_train)=='medv'))] ,s=bestlam ,type = "response"))
mean((lasso.pred_outsample-Boston_test$medv)^2)  #MSPE



################################################Residual Diagnostics and Predictive performance of the Best Model
best_model <- lm(medv~crim+zn+nox+rm+dis+rad+tax+ptratio+black+lstat, data=Boston_train)
summary(best_model)
AIC(best_model)
par(mfrow = c(2,2))
plot(best_model)


#in-sample performance of the best model
pred_train <- predict(best_model, newdata = Boston_train)
mean((pred_train-Boston_train$medv)^2) 


#out-of sample performance of the best model
pred_test <- predict(best_model, newdata = Boston_test)
mean((pred_test-Boston_test$medv)^2) 




################################################ 3: Cross Validation: 3-fold CV on the full model
boston_full <-glm(medv~., data=Boston)
cv.glm(data=Boston,glmfit = boston_full,K=3)$delta[2]  
cv.glm(data=Boston,glmfit = boston_full,K=nrow(Boston))$delta[2] #adjusted cross-validation estimate of prediction error

#3-fold Cross Validation Using Mean Absolute Error
MAE_cost <- function(pi, r){return(mean(abs(pi-r)))}
cv.glm(data = Boston, glmfit = boston_full, cost = MAE_cost, K = 3)$delta[2]




###################################################4: Regression Trees
##Full Tree
boston.rpart <- rpart(formula = medv ~ ., data = Boston_train, cp = 0.001)
boston.rpart
prp(boston.rpart,digits = 4, extra = 1)
plotcp(boston.rpart) 
printcp(boston.rpart) 

#In-sample prediction performance
boston.train.pred.tree = predict(boston.rpart)
MSE.tree<- mean((boston.train.pred.tree - Boston_train$medv)^2)
MSE.tree


#Out-of-sample prediction performance
boston.test.pred.tree = predict(boston.rpart,Boston_test)
MSPE.tree <- mean((boston.test.pred.tree - Boston_test$medv)^2)
MSPE.tree


##PRUNED TREE
boston.new.tree <- rpart(formula=medv~., data=Boston_train, cp = 0.0075200)
boston.new.tree
prp(boston.new.tree)

#In-sample prediction performance
boston.train.prune.tree = predict(boston.new.tree)
MSE.prune.tree<- mean((boston.train.prune.tree - Boston_train$medv)^2)
MSE.prune.tree


#Out-of-sample prediction performance
boston.test.prune.tree = predict(boston.new.tree,Boston_test)
MSPE.prune.tree <- mean((boston.test.prune.tree - Boston_test$medv)^2)
MSPE.prune.tree






########################################################## 5: Bagging
boston.bag<- randomForest(medv~., data = Boston_train,ntree=1000, mtry= ncol(Boston_train)-1)
boston.bag

#in-sample prediction performance
boston.bag.pred.train<- predict(boston.bag, Boston_train)
mean((Boston_train$medv-boston.bag.pred.train)^2)

#out-of-sample prediction performance 
boston.bag.pred<- predict(boston.bag, Boston_test)
mean((Boston_test$medv-boston.bag.pred)^2)


#####################out-of-bag prediction error (OOB). Goal is to observe how mse changes with number of trees
ntree<- c(1, 3, 5, seq(10, 100, 10))
MSE.test<- rep(0, length(ntree))
for(i in 1:length(ntree)){
  boston.bag<- randomForest(medv~., data = Boston_train,ntree=100, mtry= ncol(Boston_train)-1)
  boston.bag.pred<- predict(boston.bag, newdata = Boston_test)
  MSE.test[i]<- mean((Boston_test$medv-boston.bag.pred)^2)
}
plot(ntree, MSE.test, type = 'l', col=2, lwd=2, xaxt="n")
axis(1, at = ntree, las=1)

#Out-of-bag (OOB) prediction error
boston.bag$mse #mse for all number of trees
mean(boston.bag$mse)




################################################### 6: Random Forest
boston.rf<- randomForest(medv~., data = Boston_train, ntree=1000, mtry=floor(ncol(Boston_train)-1)/3)
boston.rf

#Variable importance
varImpPlot(boston.rf)

#out of bag error for different tree sizes
plot(boston.rf$mse, type='l', col=2, lwd=2, xlab = "ntree", ylab = "OOB Error")

boston.rf.pred.tr<- predict(boston.rf, Boston_train)
mean((Boston_train$medv-boston.rf.pred.tr)^2)

#out of sample prediction error
boston.rf.pred<- predict(boston.rf, Boston_test)
mean((Boston_test$medv-boston.rf.pred)^2)

#how the OOB error and testing error changes with mtry.
oob.err<- rep(0, 13)
test.err<- rep(0, 13)
for(i in 1:13){
  fit<- randomForest(medv~., data = Boston_train, mtry=i)
  oob.err[i]<- fit$mse[500]
  test.err[i]<- mean((Boston_test$medv-predict(fit, Boston_test))^2)
  cat(i, " ")
}

matplot(cbind(test.err, oob.err), pch=15, col = c("red", "blue"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("test Error", "OOB Error"), pch = 15, col = c("red", "blue"))





######################################################## 7: Boosting
boston.boost<- gbm(medv~., data = Boston_train, distribution = "gaussian", n.trees = 1000, shrinkage = 0.01, interaction.depth = 8)
summary(boston.boost)

#partial dependent plot
par(mfrow=c(1,2))
plot(boston.boost, i="lstat")
plot(boston.boost, i="rm")

#in-sample prediction performance
boston.boost.pred.train<- predict(boston.boost,Boston_train, n.trees = 1000)
mean((Boston_train$medv-boston.boost.pred.train)^2)

#out-of-sample prediction performance
boston.boost.pred.test<- predict(boston.boost, Boston_test, n.trees = 1000)
mean((Boston_test$medv-boston.boost.pred.test)^2)


#performance with varing numbers of trees
ntree<- seq(100, 1000, 100)
predmat<- predict(boston.boost, Boston_test, n.trees = ntree)
err<- apply((predmat-Boston_test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col=2, lwd=2, xlab = "n.trees", ylab = "Test MSE")
abline(h=min(test.err), lty=2)



################################################## 8: Generalized Additive Model 
###GAM Model
Boston.gam <- gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+s(age)+s(dis)+rad+s(tax)+s(ptratio)+s(black)+s(lstat),data=Boston_train)
summary(Boston.gam)
plot(Boston.gam, pages=1)

Boston.gamm <- gam(medv ~ s(crim)+s(zn)+s(indus)+chas+s(nox)+s(rm)+age+s(dis)+rad+s(tax)+ptratio+black+s(lstat),data=Boston_train)
summary(Boston.gamm)
plot(Boston.gamm, pages=1)

Boston.gam1 <- gam(medv ~ s(crim)+zn+s(indus)+chas+s(nox)+s(rm)+age+s(dis)+rad+s(tax)+ptratio+black+s(lstat),data=Boston_train)
summary(Boston.gam1)
plot(Boston.gam1, pages=1)


#Model AIC/BIC and mean residual deviance**
AIC(Boston.gam1)
BIC(Boston.gam1)
Boston.gam1$deviance

#In-sample prediction performance**
pi.train <- predict(Boston.gam1,Boston_train)
mean((pi.train - Boston_train$medv)^2)


#out of sample prediction performance
pi <- predict(Boston.gam1,Boston_test)
mean((pi - Boston_test$medv)^2)




############################################################## 9: NEURAL NETWORKS
maxs <- apply(Boston, 2, max) 
mins <- apply(Boston, 2, min)

###Standardize the data
scaled <- as.data.frame(scale(Boston, center = mins, scale = maxs - mins))
index <- sample(1:nrow(Boston),round(0.70*nrow(Boston)))
train_sc <- scaled[index,]
test_sc <- scaled[-index,]


##Build the neural network model: TWO HIDDEN LAYERS
#two hidden layers- first with 5 neurons and other with 3 neurons
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
Boston.nn <- neuralnet(f,data=train_sc,hidden=c(5,3),linear.output=T)
plot(Boston.nn)

##predictions
pr.nn <- compute(Boston.nn,test_sc[,1:13])
pr.nn_ <- pr.nn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

#in-sample prediction performance
train.r <- (test_sc$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
MSEtr.nn <- sum((train.r - pr.nn_)^2)/nrow(train_sc)
MSEtr.nn

#in-sample prediction performance
test.r <- (test_sc$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
MSEtest.nn <- sum((test.r - pr.nn_)^2)/nrow(test_sc)
MSEtest.nn




##Build the neural network model: THREEE HIDDEN LAYERS
Boston.nnn <- neuralnet(f,data=train_sc,hidden=c(5,3,6),linear.output=T)
plot(Boston.nnn)

##predictions
pr.nnn <- compute(Boston.nnn,test_sc[,1:13])
pr.nnn_ <- pr.nnn$net.result*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)

#in-sample prediction performance
train.r <- (test_sc$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
MSEtr.nnn <- sum((train.r - pr.nnn_)^2)/nrow(train_sc)
MSEtr.nn

#in-sample prediction performance
test.r <- (test_sc$medv)*(max(Boston$medv)-min(Boston$medv))+min(Boston$medv)
MSEtest.nnn <- sum((test.r - pr.nnn_)^2)/nrow(test_sc)
MSEtest.nnn


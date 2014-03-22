library('ggplot2')
library('Cubist')
library(caret)
library(C50)
library(psych)
library(gbm)
library(doMC)
library('randomForest')
library(ROCR)
load("~/uncorreled_preds.rda")
library(caretEnsemble)

#chnage the following line: mse <- mean(abs(exp(pred) - exp(obs)))
trace("postResample",edit=TRUE)

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]

#add new gloden feats
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271

keep_names <- names(train.surv)
save(keep_names, file = "keep_names.rda")

#keep_cols <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405','f405','f599','f1')
#ain.surv) %in% keep_cols)]
#sapply(train.surv, function(x)all(!is.na(x)))
gc()
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
save(preProc, file = "preprocess_center_scale_impute.rda")
train.reg <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
gc()
train.reg$loss <- train.surv$loss
gc()

train.reg <- train[,-which(names(train) %in% c('returned'))]
train.reg$loss <- train.loss
train.reg <- train.reg[train.reg$loss>0,]
train.reg <- na.omit(train.reg)
set.aside <- train.reg$loss
train.reg$loss <- log(train.reg$loss)
gbmGrid = expand.grid(.interaction.depth = c(25) , .n.trees = c(1200), .shrinkage = c(.1))
rfGrid = expand.grid(.mtry = c(3) )
gc()

#create holdout for calibration
#samp.indx <- sample(c(1:nrow(train.reg)),1000)
#test.reg <- train.reg[samp.indx,]
#train.reg <- train.reg[-samp.indx,]



####################
#
### bagging validation for just rgression
#
#####################

library('ggplot2')
library('Cubist')
library(caret)
library(C50)
library(psych)
library(gbm)
library(doMC)
library('randomForest')

#chnage the following line: mse <- mean(abs(exp(pred) - exp(obs)))
trace("postResample",edit=TRUE)

#load training set
load("~/uncorreled_preds.rda")
remove_preds <- c("f662","f663","f159","f160","f169","f170","f618","f619","f330","f331","f179","f180","f422","f653","f189","f190","f340",
                        "f341","f664","f665","f666","f667","f668","f669","f726","f640","f199","f200","f650","f651","f72","f586","f587","f649",
                        "f648","f588","f620","f621","f672","f673","f209","f210","f679","f149","f150","f32","f142","f143","f151","f152","f161",
                        "f162","f171","f172","f181","f182","f191","f192","f201","f202","f393","f394","f45","f46")
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols <- c('f670','f67',uncorrelated_preds)
keep_cols <- keep_cols[!(keep_cols %in% remove_preds)]
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)
library(foreach)
results <-  foreach(i=1:10, .combine='rbind', .packages=c('Cubist','gbm')) %dopar% {
  # get splits:
  samp.indx <- sample(c(1:nrow(train.reg)),1500)
  test.set <- train.reg[samp.indx,]
  train.set <- train.reg[-samp.indx,]
  
  gbm_fit_prop <- gbm(formula = loss~.,
                      distribution = "laplace",
                      data = train.set,
                      n.trees = 1200,
                      interaction.depth = 40,
                      shrinkage = 0.04)
  
  cubfit_reg_prop <- cubist(y = train.set$loss, x = train.set[,-which(names(train.set) %in% c('loss'))],
                            committees = 20)
  
  test.me <- function(mod,mod2,alpha){
    preds <- predict(mod,test.set[,-which(names(test.set) %in% c('loss'))],n.trees = 1200)
    preds2 <- predict(mod2,test.set[,-which(names(test.set) %in% c('loss'))])
    preds <- (alpha*preds+(1-alpha)*preds2)
    res <- mean(abs(exp(test.set$loss)-exp(preds)))
    return(res)
  }
  
  derp <- sapply(seq(0,1,.02),function(x) test.me(gbm_fit_prop,cubfit_reg_prop,x))
  (opt.blend <- seq(0,1,.02)[which(derp==min(derp))])
  plot(seq(0,1,.02),derp)
  opt.blend_mid <- derp[which(seq(0,1,.02)==.62)]
  c(opt.blend,min(derp),opt.blend_mid)
}




####################
#
### bagging for quantile regression
#
#####################

library('ggplot2')
library('Cubist')
library(caret)
library(C50)
library(psych)
library(gbm)
library(doMC)
library('randomForest')

#chnage the following line: mse <- mean(abs(exp(pred) - exp(obs)))
trace("postResample",edit=TRUE)

#load training set
load("~/uncorreled_preds.rda")
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
#load training set
load("~/uncorreled_preds.rda")
remove_preds <- c("f662","f663","f159","f160","f169","f170","f618","f619","f330","f331","f179","f180","f422","f653","f189","f190","f340",
                  "f341","f664","f665","f666","f667","f668","f669","f726","f640","f199","f200","f650","f651","f72","f586","f587","f649",
                  "f648","f588","f620","f621","f672","f673","f209","f210","f679","f149","f150","f32","f142","f143","f151","f152","f161",
                  "f162","f171","f172","f181","f182","f191","f192","f201","f202","f393","f394","f45","f46")
keep_cols <- c('f670','f67',uncorrelated_preds)
keep_cols <- keep_cols[!(keep_cols %in% remove_preds)]
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)
library(foreach)
registerDoMC(cores = 6)

results <-  foreach(i=1:10, .combine='rbind', .packages=c('Cubist','gbm')) %dopar% {
  # get splits:
  samp.indx <- sample(c(1:nrow(train.reg)),nrow(train.reg)/3)
  test.set <- train.reg[samp.indx[c((length(samp.indx)/2+1):length(samp.indx))],]
  valid.set <- train.reg[samp.indx[c(1:(length(samp.indx)/2))],]
  #valid.set <- train.reg[samp.indx,]
  train.set <- train.reg[-samp.indx,]
  
  gbm_regular<- gbm(formula = loss~.,
                           distribution = 'laplace',
                           data = train.set,
                           n.trees = 1100,
                           interaction.depth = 24,
                           shrinkage = 0.05)
  
  gbm_fit_lowerquant<- gbm(formula = loss~.,
                      distribution = list(name="quantile",alpha=0.15),
                      data = train.set,
                      n.trees = 1100,
                      interaction.depth = 24,
                      shrinkage = 0.05)
  
  gbm_fit_midquant <- gbm(formula = loss~.,
                      distribution = list(name="quantile",alpha=0.50),
                      data = train.set,
                      n.trees = 1100,
                      interaction.depth = 24,
                      shrinkage = 0.05)
  
  gbm_fit_upperquant <- gbm(formula = loss~.,
                      distribution = list(name="quantile",alpha=0.85),
                      data = train.set,
                      n.trees = 1100,
                      interaction.depth = 24,
                      shrinkage = 0.05)
  
  quantbag <- data.frame(f670 = valid.set$f670,
                         f67 = valid.set$f67,
                        reg = predict(gbm_regular,valid.set,n.trees = 1100),
                        lower = predict(gbm_fit_lowerquant,valid.set,n.trees = 1100),
                         middle = predict(gbm_fit_midquant,valid.set,n.trees = 1100),
                         upper = predict(gbm_fit_upperquant,valid.set,n.trees = 1100),
                         loss = valid.set$loss)
  
  #gbm_fit_prop <- gbm(formula = loss~.,
  #                    distribution = "laplace",
  #                    data = quantbag,
  #                    n.trees = 500,
  #                    interaction.depth = 6,
  #                    shrinkage = 0.1)
  
  rf_fit_prop <- randomForest(formula = loss~.,data = quantbag)
  
  #save(gbm_regular,gbm_fit_lowerquant,gbm_fit_midquant,gbm_fit_upperquant,rf_fit_prop,file="big_t4p5_regression.rda")
  
  
  test.quantbag <- data.frame(f670 = test.set$f670,
                              f67 = test.set$f67,
                              reg = predict(gbm_regular,test.set,n.trees = 1100),
                        lower = predict(gbm_fit_lowerquant,test.set,n.trees = 1100),
                         middle = predict(gbm_fit_midquant,test.set,n.trees = 1100),
                         upper = predict(gbm_fit_upperquant,test.set,n.trees = 1100),
                         loss = test.set$loss)
  
  
  
  all_mean <- cbind(predict(gbm_fit_lowerquant,test.set,n.trees = 1100),
                    predict(gbm_fit_midquant,test.set,n.trees = 1100),
                    predict(gbm_fit_upperquant,test.set,n.trees = 1100))
  all_mean <- exp(rowMeans(all_mean))
  
  ret <- exp(predict(rf_fit_prop,test.quantbag))
  reslower <- mean(abs(exp(predict(gbm_fit_lowerquant,test.set,n.trees = 1100))-exp(test.set$loss)))
  resmid <- mean(abs(exp(predict(gbm_fit_midquant,test.set,n.trees = 1100))-exp(test.set$loss)))
  resupper <- mean(abs(exp(predict(gbm_fit_upperquant,test.set,n.trees = 1100))-exp((test.set$loss))))
  resbag <- mean(abs(ret-exp(test.set$loss)))
  rereg <- mean(abs(exp(predict(gbm_regular,test.set,n.trees = 1100))-exp((test.set$loss))))
  meanreg <- mean(abs(all_mean-exp((test.set$loss))))
  
  c(rereg,meanreg,reslower,resmid,resupper,resbag)
}

apply_quantile<- function(test.set){
  
  load('~/big_t4p5_regression.rda')
  
  test.quantbag <- data.frame(f670 = test.set$f670,
                              f67 = test.set$f67,
                              f377 = test.set$f377,
                              reg = predict(gbm_regular,test.set,n.trees = 1000),
                              lower = predict(gbm_fit_lowerquant,test.set,n.trees = 1000),
                              middle = predict(gbm_fit_midquant,test.set,n.trees = 1000),
                              upper = predict(gbm_fit_upperquant,test.set,n.trees = 1000),
                              loss = test.set$loss)
  
  ret <- exp(predict(rf_fit_prop,test.quantbag))
  return(ret)
}


####################
#
### bagging validation for the classifier
#
#####################

#F1 BEST SO FAR:
# 0.9479987, cutoff = 0.4727547 with tier_1 + 232
# 0.9513712, cutoff = 0.5163318 with tier_2 + 232
# 0.9487212, cutoff = 0.5169211 with tier_1 + 232 + 222
# 0.9557908, cutoff = 0.5018422 median = 0.4784802 with tier_4
#gbm with weights seems to suck... 

registerDoMC(cores = 6)

results <-  foreach(i=1:10, .combine='rbind', .packages=c('gbm','ROCR')) %dopar% {
  # get splits:
  samp.indx <- sample(c(1:nrow(train)),round(nrow(train)/5))
  test.prop <- na.omit(train[samp.indx,])
  #test.prop$returned <- test.prop$loss
  train.prop <- na.omit(train[-samp.indx,])
  #weights.gbm <- ifelse(train.prop$returned==1,2,1)
  #train.prop$returned <- as.numeric(as.character(train.prop$returned))
  
  gbm_fit_prop <- gbm(formula = returned~.,
                      distribution = "bernoulli",
                      #weights = weights.gbm,
                      data = train.prop,
                      n.trees = 1100,
                      interaction.depth = 14,
                      shrinkage = 0.05)
  
  
  
  preds <- plogis(predict(gbm_fit_prop,test.prop,n.trees = 1100))
  pred <- prediction(preds, test.prop$returned)
  f <- performance(pred, 'f')
  f1_score <- f@y.values[[1]]
  cutoff <- f@x.values[[1]]
  best_f1_score <- max(f1_score,na.rm=T)
  best_cutoff <- cutoff[which.max(f1_score)]
  
  c(best_f1_score,best_cutoff)
}



####################
#
### train some stuff
#
#####################

#tier 0 model

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols <- c("yr","yr2","yr3","f2","f271","f670","f67","loss")
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc_0 <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc_0,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)

gbmGrid = expand.grid(.interaction.depth = c(4) , .n.trees = c(1100), .shrinkage = c(.04))
gbm_reg_t0 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))


save(gbm_reg_t0, file = "gbm_reg_t0.rda")
capture.output(gbm_reg_t0, file = "gbm_reg_t0.txt")

#tier 1 model

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols <- c("yr","yr2","yr3","f2","f271","f670","f67","loss")
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)

gbmGrid = expand.grid(.interaction.depth = c(6) , .n.trees = c(1100), .shrinkage = c(.04))
gbm_reg_t1 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))

gbm_reg

save(gbm_reg_t1, file = "gbm_reg_t1.rda")
capture.output(gbm_reg_t1, file = "gbm_reg_t1.txt")

cubreg_reg_t1 <- train(loss~.,data=train.reg,
                       "cubist", 
                       trControl = trainControl(method = "cv"))
save(cubreg_reg_t1, file = "cubreg_reg_t1.rda")
capture.output(cubreg_reg_t1, file = "cubreg_reg_t1.txt")

gbm_reg_t15 <- train(loss~.,data=train.reg,
                     "gbm", 
                     distribution = 'laplace',
                     tuneGrid = gbmGrid,
                     trControl = trainControl(method = "cv"))

gbm_reg

save(gbm_reg_t1, file = "gbm_reg_t1.rda")
capture.output(gbm_reg_t1, file = "gbm_reg_t1.txt")


#tier 2 model

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
#keep_cols <- tier_2 <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405','f599','f1')
keep_cols <- c('f527','f274','f271','yr','yr2','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f282','f596','f377','f9','f278','f273','f406','f599','f1','f130','f514')
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- train.reg[train.surv$loss >0,]
train.reg <- na.omit(train.reg)
X <- train.reg[,-which(names(train.reg) %in% c('loss'))]
Y <- log(train.reg$loss)

#test train split
trainsplit <- runif(nrow(X)) <= .85
registerDoMC(cores = 6)
folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, returnResamp='none', 
                          returnData=FALSE, savePredictions=TRUE, 
                          verboseIter=TRUE, allowParallel=TRUE,
                          index=createMultiFolds(Y[trainsplit], k=folds, times=repeats))

model1 <- train(X[trainsplit,], Y[trainsplit], method='gbm', trControl=myControl,distribution = 'laplace',
                tuneGrid=expand.grid(.n.trees=1000, .interaction.depth=6,.shrinkage = c(.05)))
model2 <- train(X[trainsplit,], Y[trainsplit], method='glmnet',tuneGrid=expand.grid(.alpha=c(.1,.4,.85),.lambda = c(0.1,0.4,1)), trControl=myControl)
model3 <- train(X[trainsplit,], Y[trainsplit], method='svmRadial', trControl=myControl)
model4 <- train(X[trainsplit,], Y[trainsplit], method='cubist', tuneGrid=expand.grid(.committees=c(20,30),.neighbors = 0),trControl=myControl)
model5 <- train(X[trainsplit,], Y[trainsplit], method='ppr', trControl=myControl)
model6 <- train(X[trainsplit,], Y[trainsplit], method='rf',tuneGrid=expand.grid(.mtry=c(3,5)) , trControl=myControl)

all.models <- list(model1, model2, model3, model4, model5,model6)
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$RMSE)))

#create a greedy ensemble
greedy <- caretEnsemble(all.models, iter=1000L)
sort(greedy$weights, decreasing=TRUE)
greedy$error

#create glm ensemble
linear_t2 <- caretStack(all.models, method='glm', trControl=trainControl(method='cv'))
summary(linear_t2$ens_model$finalModel)
linear_t2$error

# test set error
preds <- data.frame(sapply(all.models, predict, newdata=X[!trainsplit,]))
preds$ENS_greedy <- predict(greedy, newdata=X[!trainsplit,])
preds$ENS_linear <- predict(linear_t2, newdata=X[!trainsplit,])
sort(colMeans(abs(exp(preds) - exp(Y[!trainsplit]))))

save(linear_t2, file = "ensemble_t2.rda")

gbmGrid = expand.grid(.interaction.depth = c(14) , .n.trees = c(1100), .shrinkage = c(.04))
gbm_reg_t2 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))


save(gbm_reg_t2, file = "gbm_reg_t2.rda")
capture.output(gbm_reg_t2, file = "gbm_reg_t2.txt")

cubreg_reg_t2 <- train(loss~.,data=train.reg,
                       "cubist", 
                       tuneGrid=expand.grid(.committees=c(20,30),.neighbors = 0),
                       trControl = trainControl(method = "cv"))
save(cubreg_reg_t2, file = "cubreg_reg_t2.rda")
capture.output(cubreg_reg_t2, file = "cubreg_reg_t2.txt")

#############
##### 3
##### 3
#tier 3 model
##### 3
##### 3
#############

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols <- c(keep_cols_reg,keep_cols_classification)
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)

gbmGrid = expand.grid(.interaction.depth = c(16) , .n.trees = c(1200), .shrinkage = c(.04))
gbm_reg_t3 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))

save(gbm_reg_t3, file = "gbm_reg_t3.rda")
capture.output(gbm_reg_t3, file = "gbm_reg_t3.txt")

cubreg_reg_t3 <- train(loss~.,data=train.reg,
                       "cubist", 
                       tuneGrid=expand.grid(.committees=c(20,30),.neighbors = 0),
                       trControl = trainControl(method = "cv"))
save(cubreg_reg_t3, file = "cubreg_reg_t3.rda")
capture.output(cubreg_reg_t3, file = "cubreg_reg_t3.txt")


#############
##### 4
##### 4
#tier 4 model
##### 4
##### 4
#############
#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
keep_cols <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer)
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- na.omit(train.reg)
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train$loss <- ifelse(train.surv$loss >0,1,0)

gbmGrid = expand.grid(.interaction.depth = c(12,16,20) , .n.trees = c(1200), .shrinkage = c(.04))
gbm_reg_t4 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))


capture.output(gbm_reg_t4, file = "gbm_reg_t4.txt")

cubreg_reg_t4 <- train(loss~.,data=train.reg,
                       "cubist", 
                       trControl = trainControl(method = "cv"))
save(cubreg_reg_t4,gbm_reg_t4, file = "reg_t4_opt2.rda")
capture.output(cubreg_reg_t4, file = "cubreg_reg_t4.txt")

svmGrid = expand.grid(.C = c(1) , .degree = c(2,3), .scale = c(.001))
svmpoly_reg_t4 <- svmPoly_reg
svmpoly_reg_t4 <- train(loss~.,data=train.reg,
                       "svmPoly", 
                       tuneGrid = svmGrid,
                       trControl = trainControl(method = "cv"))
save(svmpoly_reg_t4, file = "svmpoly_reg_t4.rda")
capture.output(svmpoly_reg_t4, file = "svmpoly_reg_t4.txt")

##### bagging on top:
train.bag <- data.frame(gbm_mod = predict(gbm_reg_t4,train.reg),
           cubistmod_mod = predict(cubreg_reg_t4,train.reg),
           svm_mod = predict(svmpoly_reg_t4,train.reg),
           predicted_loss = round(plogis(predict(gbmfit$finalModel,train.reg,n.trees=1000)),3),
           loss = train.reg$loss)

glmboost_grid = expand.grid(.mstop=c(3:5)*50,.prune=('no'))
train_bag_t4 <- train(loss~.,data=train.bag,
                        "rf", 
                        tuneGrid = expand.grid(.mtry=c(2,3)),
                        trControl = trainControl(method = "cv"))
save(train_bag_t4, file = "train_bag_t4.rda")
capture.output(train_bag_t4, file = "train_bag_t4.txt")

#################
###### 4.5
###### 4.5
#tier 4.5 model

set.seed(42) #From random.org
library(caretEnsemble)

train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
p5_new <- c("f598","f596","f431","f3","f71","f673","f212","f271","f640","f746","f16","","f432","f413","yr","f601","f406","f775","f433","f132","f75","f655","f609","f614","f208","f522","f733","f774","f377","f518","f19","f6","f384","f422","f533","f367","f366","f273","f282","f647","f756","f57","f361","f631","f76","f1","f130","f652")
keep_cols <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer,p5_new)
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
#preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss'))],method = c("center", "scale"),verbose=TRUE)
#train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- train.reg[train.surv$loss >0,]
train.reg$loss <- log(train.reg$loss)
#train.reg <- na.omit(train.reg)
X <- train.reg[,-which(names(train.reg) %in% c('loss'))]
Y <- train.reg$loss


preProc_knnimp <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss','id'))],
                      method = c("center", "scale",'knnImpute'),
                      knnSummary = median,
                      verbose=TRUE)
train <- predict(preProc_knnimp,train.surv[train.surv$loss >0,-which(names(train.surv) %in% c('loss','id'))])
train$loss <- log(train.surv$loss[train.surv$loss >0])

registerDoMC(cores = 5)
gbmGrid = expand.grid(.interaction.depth = c(20) , .n.trees = c(1100), .shrinkage = c(.05))
gbm_reg_impute4.5 <- train(loss~.,data=train,
                      "gbm", 
                      distribution = 'laplace',
                      tuneGrid = gbmGrid,
                      trControl = trainControl(method = "cv"))

save(preProc_knnimp,gbm_reg_impute4.5, file = "imputed_t4p5.rda")


#test train split
trainsplit <- runif(nrow(X)) <= .80
registerDoMC(cores = 6)
folds=5
repeats=1
myControl <- trainControl(method='cv', number=folds, repeats=repeats, returnResamp='none', 
                          returnData=FALSE, savePredictions=TRUE, 
                          verboseIter=TRUE, allowParallel=TRUE,
                          index=createMultiFolds(Y[trainsplit], k=folds, times=repeats))

model1 <- train(X[trainsplit,], Y[trainsplit], method='gbm', trControl=myControl,distribution = 'laplace',
                tuneGrid=expand.grid(.n.trees=1000, .interaction.depth=26,.shrinkage = c(.05)))
model2 <- train(X[trainsplit,], Y[trainsplit], method='glmnet',tuneGrid=expand.grid(.alpha=c(.1,.4,.85),.lambda = c(0.1,0.4,1)), trControl=myControl)
model3 <- train(X[trainsplit,], Y[trainsplit], method='svmRadial', trControl=myControl)
model4 <- train(X[trainsplit,], Y[trainsplit], method='cubist', trControl=myControl)
model5 <- train(X[trainsplit,], Y[trainsplit], method='ppr', trControl=myControl)
model6 <- train(X[trainsplit,], Y[trainsplit], method='rf',tuneGrid=expand.grid(.mtry=c(5,12)), trControl=myControl)

all.models <- list(model1, model2, model3, model4, model5,model6)
names(all.models) <- sapply(all.models, function(x) x$method)
sort(sapply(all.models, function(x) min(x$results$RMSE)))

#create a greedy ensemble
greedy <- caretEnsemble(all.models, iter=1000L)
sort(greedy$weights, decreasing=TRUE)
greedy$error

#create rf ensemble
linear <- caretStack(all.models, method='glm', trControl=trainControl(method='cv'))
summary(linear$ens_model$finalModel)
linear$error

# test set error
preds <- data.frame(sapply(all.models, predict, newdata=X[!trainsplit,]))
preds$ENS_greedy <- predict(greedy, newdata=X[!trainsplit,])
preds$ENS_linear <- predict(linear, newdata=X[!trainsplit,])
sort(colMeans(abs(preds - Y[!trainsplit])))

save(all.models,linear, file = "ensemble_t4p5.rda")



registerDoMC(cores = 6)

gbmGrid = expand.grid(.interaction.depth = c(20) , .n.trees = c(1100), .shrinkage = c(.05))
gbm_reg_t4p5 <- train(loss~.,data=train.reg,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))

save(gbm_reg_t4p5, file = "gbm_reg_t4p5.rda")
capture.output(gbm_reg_t4p5, file = "gbm_reg_t4p5.txt")

cubreg_reg_t4p5 <- train(loss~.,data=train.reg,
                       "cubist", 
                       trControl = trainControl(method = "cv"))
save(cubreg_reg_t4p5, file = "cubreg_reg_t4p5.rda")
capture.output(cubreg_reg_t4p5, file = "cubreg_reg_t4p5.txt")






#tier 5 model

#load training set
train.surv <- read.csv("~/Documents/Personal/kaggle/loans/train.csv",stringsAsFactors=FALSE)
train.surv<- train.surv[,unlist(lapply(train.surv, function(x) length(table(x))))>1]
train.surv$yr <- train.surv$f528 - train.surv$f527
train.surv$yr2 <- train.surv$f528 - train.surv$f274
train.surv$yr3 <- train.surv$f528 - train.surv$f271
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
keep_cols <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer,uncorrelated_preds)
train.surv  <- train.surv[,which(names(train.surv) %in% keep_cols)]
####
#### temp section
indx <-sample(c(1:nrow(train.surv)),nrow(train.surv)/4)
test.surv <- train.surv[indx,]
train.surv <- train.surv[-indx,]

preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss','id'))],
                      method = c("center", "scale",'knnImpute'),
                      knnSummary = median,
                      verbose=TRUE)
train <- predict(preProc,train.surv[train.surv$loss >0,-which(names(train.surv) %in% c('loss','id'))])
test <- predict(preProc,test.surv[test.surv$loss>0,-which(names(test.surv) %in% c('loss','id'))])
train$loss <- log(train.surv$loss[train.surv$loss >0])
test$loss <- test.surv$loss[test.surv$loss >0]
preds <- predict(gbm_reg_t4p5,test)
test$preds <-preds
mae <- mean(abs(exp(test$preds)- test$loss[test$loss>0]))




preProc <- preProcess(train.surv[,-which(names(train.surv) %in% c('loss','id'))],method = c("center", "scale"),verbose=TRUE)
train <- predict(preProc,train.surv[,-which(names(train.surv) %in% c('loss','id'))])
train.reg <- train
train.reg$loss <- train.surv$loss
train.reg <- train.reg[train.reg$loss >0,]
train.reg$loss <- log(train.reg$loss)
train.reg <- na.omit(train.reg)
#train$loss <- ifelse(train.surv$loss >0,1,0)


registerDoMC(cores = 5)

gbmGrid = expand.grid(.interaction.depth = c(20,40) , .n.trees = c(1200), .shrinkage = c(.04))
gbm_reg_t5 <- train(loss~.,data=train,
                    "gbm", 
                    distribution = 'laplace',
                    tuneGrid = gbmGrid,
                    trControl = trainControl(method = "cv"))

save(gbm_reg_t5, file = "gbm_reg_t5.rda")
capture.output(gbm_reg_t5, file = "gbm_reg_t5.txt")

glmboost_grid = expand.grid(.mstop = c(4:8)*100,.prune=c("no","yes"))

cubreg_reg_t5 <- train(loss~.,data=train.reg,
                       "cubist", 
                       trControl = trainControl(method = "cv"))

save(cubreg_reg_t5, file = "cubreg_reg_t5.rda")
capture.output(cubreg_reg_t5, file = "cubreg_reg_t5.txt")


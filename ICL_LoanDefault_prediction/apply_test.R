library('ggplot2')
library('Cubist')
library(caret)
library(C50)
library(psych)
library(gbm)
library(doMC)
library('randomForest')
library(ROCR)
library(caretEnsemble)


# load necessary models
load("~/gbm_classifier.rda")
load("~/gbm_reg_t4.rda")
load("~/cubreg_reg_t4.rda")
load("~/svmpoly_reg_t4.rda")
load('~/train_bag_t4.rda')
load("~/gbm_reg_t3.rda")
load("~/cubreg_reg_t3.rda")
load("~/gbm_reg_t2.rda")
load("~/cubreg_reg_t2.rda")
load("~/gbm_reg_t1.rda")
load("~/cubreg_reg_t1.rda")
load("~/preprocess_center_scale_impute.rda")
load("~/keep_names.rda")
load("~/gbm_classifier_t4.rda")
load("~/gbm_reg_t5.rda")
load("~/cubreg_reg_t5.rda")
load("~/gbm_reg_t4p5.rda")
load("~/cubreg_reg_t4p5.rda")
load("~/uncorreled_preds.rda")
load("~/gbm_reg_t0.rda")
load("~/ensemble_t2.rda")
load("~/imputed_t4p5.rda")


# attribute sets to keep

tier_0 <- c("yr","yr2","yr3","f2","f271","f670","f67","loss")
#t1
tier_1 <- c("yr","yr2","yr3","f2","f271","f670","f67","loss","f514","f598")
#t2
tier_2 <- c('f527','f274','f271','yr','yr2','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f282','f596','f377','f9','f278','f273','f406','f599','f1','f130','f514')
#t3
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f406',"returned",'f655','f130','f739')
tier_3 <- c(keep_cols_reg,keep_cols_classification)
#t4
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
tier_4 <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer)

#t4.5
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
p5_new <- c("f598","f596","f431","f3","f71","f673","f212","f271","f640","f746","f16","","f432","f413","yr","f601","f406","f775","f433","f132","f75","f655","f609","f614","f208","f522","f733","f774","f377","f518","f19","f6","f384","f422","f533","f367","f366","f273","f282","f647","f756","f57","f361","f631","f76","f1","f130","f652")
tier_4.5 <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer,p5_new)
#t5
keep_cols_reg <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f71','f9','f278','f273','f405','f405','f514','f431','f3','f640','f596','f746','f413','f673',"returned")
keep_cols_classification  <- c('f471','f527','f274','f271','yr','yr2','yr3','f528',"loss",'f670','f322','f404','f766','f67','f281','f13','f598','f68','f282','f2','f282','f767','f377','f9','f278','f273','f405',"returned")
keep_cols_newer <- c('f133','f739','f132','f432','f655','f16','f518','f384','f281','f775','f522','f669','f406','f142','f282','f75')
tier_5 <- c(keep_cols_reg,keep_cols_classification,keep_cols_newer,uncorrelated_preds)

#load training set
test.surv <- read.csv("~/Documents/Personal/kaggle/loans/test.csv",stringsAsFactors=FALSE)
test.surv$yr <- test.surv$f528 - test.surv$f527
test.surv$yr2 <- test.surv$f528 - test.surv$f274
test.surv$yr3 <- test.surv$f528 - test.surv$f271
#save ids
test.ids <- test.surv$id
test.surv  <- test.surv[,which(names(test.surv) %in% keep_names)]
#apply scaling
test <- predict(preProc,test.surv)
#get all missing indecies
missing.indecies_overall <- which(apply(test[,which(names(test) %in% tier_0)], 1, function(x) !all(!is.na(x))))
alive.indecies_overall <- which(apply(test[,which(names(test) %in% tier_0)], 1, function(x) all(!is.na(x))))
  #tier1
missing.indecies_t1 <- which(apply(test[,which(names(test) %in% tier_1)], 1, function(x) !all(!is.na(x))))
alive.indecies_t1 <- which(apply(test[,which(names(test) %in% tier_1)], 1, function(x) all(!is.na(x))))
  #tier2
  missing.indecies_t2 <- which(apply(test[,which(names(test) %in% tier_2)], 1, function(x) !all(!is.na(x))))
  alive.indecies_t2 <- which(apply(test[,which(names(test) %in% tier_2)], 1, function(x) all(!is.na(x))))
  #tier3
  missing.indecies_t3 <- which(apply(test[,which(names(test) %in% tier_3)], 1, function(x) !all(!is.na(x))))
  alive.indecies_t3 <- which(apply(test[,which(names(test) %in% tier_3)], 1, function(x) all(!is.na(x))))
  #tier4
  missing.indecies_t4 <- which(apply(test[,which(names(test) %in% tier_4)], 1, function(x) !all(!is.na(x))))
  alive.indecies_t4 <- which(apply(test[,which(names(test) %in% tier_4)], 1, function(x) all(!is.na(x))))
  #tier4.5
  missing.indecies_t4.5 <- which(apply(test[,which(names(test) %in% tier_4.5)], 1, function(x) !all(!is.na(x))))
  alive.indecies_t4.5 <- which(apply(test[,which(names(test) %in% tier_4.5)], 1, function(x) all(!is.na(x))))
  #tier5
  missing.indecies_t5 <- which(apply(test[,which(names(test) %in% tier_5)], 1, function(x) !all(!is.na(x))))
  alive.indecies_t5 <- which(apply(test[,which(names(test) %in% tier_5)], 1, function(x) all(!is.na(x))))

#preds <- ifelse(plogis(predict(gbmfit$finalModel,newdata = test[alive.indecies_overall,],n.trees = 1000))<0.4858394,1,0)
preds <- ifelse(as.numeric(as.character(predict(gbmfit,newdata = test[alive.indecies_overall,])))>0,1,0)
alive.vals <- data.frame(indx = alive.indecies_overall,loss = preds)
#alive.vals$loss[alive.vals$indx %in% alive.indecies_t4] <- 
#  ifelse(as.numeric(as.character(predict(gbmfit_t4,newdata = test[alive.indecies_t4,])))>0,1,0)


##################
####
#   make teir predictions
####
##################
test_t0 <- test[alive.indecies_overall,which(names(test) %in% tier_0)]
alive.vals.reg <- exp(predict(gbm_reg_t0,test_t0))

test_t1 <- test[alive.indecies_t1,]
alive.vals.reg_t1  <- exp(predict(gbm_reg_t1,test_t1)))
test_t2 <- test[alive.indecies_t2,which(names(test) %in% tier_2)]
alive.vals.reg_t2 <- exp(predict(gbm_reg_t2,test_t2))*.60+ predict(cubreg_reg_t2,test_t2)*.40
  
  #exp(predict(gbm_reg_t2,test_t2)*.60
  #                    + predict(cubreg_reg_t2,test_t2)*.40)


test_t3 <- test[alive.indecies_t3,which(names(test) %in% tier_3)]
alive.vals.reg_t3 <- exp(predict(gbm_reg_t3,test_t3)*.60
                      + predict(cubreg_reg_t3,test_t3)*.40)

test_t4 <- test[alive.indecies_t4,which(names(test) %in% tier_4)]
#test.bag_t4 <- data.frame(gbm_mod = predict(gbm_reg_t4,test_t4),
#                        cubistmod_mod = predict(cubreg_reg_t4,test_t4),
#                        svm_mod = predict(svmpoly_reg_t4,test_t4),
#                        predicted_loss = predict(gbmfit$finalModel,test_t4,n.trees=1000))
#alive.vals.reg_t4 <- exp(predict(train_bag_t4,test.bag_t4))
alive.vals.reg_t4 <- exp(predict(gbm_reg_t4,test_t4)*.60
                         + predict(cubreg_reg_t4,test_t4)*.40)

test_t4.5 <- test[alive.indecies_t4.5,which(names(test) %in% tier_4.5)]
gc()
#test.bag_t4 <- data.frame(gbm_mod = predict(gbm_reg_t4,test_t4),
#                        cubistmod_mod = predict(cubreg_reg_t4,test_t4),
#                        svm_mod = predict(svmpoly_reg_t4,test_t4),
#                        predicted_loss = predict(gbmfit$finalModel,test_t4,n.trees=1000))
#alive.vals.reg_t4 <- exp(predict(train_bag_t4,test.bag_t4))
alive.vals.reg_t4.5 <- exp(predict(gbm_reg_t4p5,test_t4.5)*.2
                         + predict(cubreg_reg_t4p5,test_t4.5)*.8)

val <- round(nrow(test_t4.5)/2)
#pred1 <- predict(linear_t4.5,test_t4.5[1:val,])
#gc()
#pred2 <- predict(linear_t4,test_t4.5[(val+1):nrow(test_t4.5),])
#gc()
#alive.vals.reg_t4.5 <- c(pred1,pred2)

test_t5 <- test[alive.indecies_t5,]
alive.vals.reg_t5 <- exp(predict(gbm_reg_t5,test_t5)*.2 #should be closer to .35
                         + predict(cubreg_reg_t5,test_t5)*.8)

imputed_t4.5 <- predict(preProc_knnimp,test.surv[alive.vals$indx[alive.vals$loss >0],which(names(test) %in% tier_4.5)])
alive.vals.impute_t4.5 <- exp(predict(gbm_reg_impute4.5,imputed_t4.5))


##################
####
#   apply predictions
####
##################

alive.vals$loss <- ifelse(alive.vals$loss>0,alive.vals.reg,0)
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t1] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t1] >0,alive.vals.reg_t1,0)
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t2] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t2] >0,alive.vals.reg_t2,0)
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t3] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t3] >0,alive.vals.reg_t3,0)
#alive.vals$loss <- alive.vals.impute_t4.5
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t4] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t4] >0,alive.vals.reg_t4,0)
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t4.5] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t4.5] >0,alive.vals.reg_t4.5,0)
alive.vals$loss[alive.indecies_overall %in% alive.indecies_t5] <- 
  ifelse(alive.vals$loss[alive.indecies_overall %in% alive.indecies_t5] >0,alive.vals.reg_t5,0)

table(ifelse(alive.vals$loss>0,1,0),preds)

alive.vals$loss <- ifelse(alive.vals$loss>100,100,alive.vals$loss)
missing.vals <- data.frame(indx = missing.indecies_overall, loss = rep(0,length(missing.indecies_overall)))
all.vals <- rbind(alive.vals,missing.vals)
all.vals <- all.vals[order(all.vals$indx),]
all.vals$id <- test.ids

res<- data.frame(id=all.vals$id,loss=all.vals$loss)
write.table(res, file="sixtier_oneclass_t2expgbm_flatcubist.csv", row.names=FALSE, sep=",")



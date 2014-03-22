Kmeans++ in Julia
--------
Adrian Haldenby a.haldenby@gmail.com

Repository for finished Kaggle competitions code. 


**Loan Default Prediction - Imperial College London**
Placed: 24/677
train.R trains the binary default/did not default classifier with a gbm and tests the f1 score, It also created a series of loss given default models again using gbms with absolute error.
apply_test.R loads the models generated in train.R and apples them to the test dat sets

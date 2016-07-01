# Prediction Assignment Writeup




## Executive Summary
The goal of this project is to predict the manner in which they did the exercise with the data given in the training data set. After reading and clearning data, the training data set was seperated into training and validation dataset. Then 3 different algorithms were used for modelling and the results were analyzed in terms of performance. Best models were selected, based on maximum accuracy, and used to predict the training set. 

### set seed and load packages

```r
set.seed(07012016)
library(caret)
library(plyr)
```

### data cleaning
Read training and testing dataset. Since some variables like time has nothing to do with the prediction, those columns were deleted from dataset

```r
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainset <- read.csv(url(trainUrl), na.strings = c("NA", "#DIV/0!", ""))
dim(trainset)
```

```
## [1] 19622   160
```

```r
trainset <- trainset[,-c(1:7)]

testset <- read.csv(url(testUrl), na.strings = c("NA", "#DIV/0!", ""))
dim(testset)
```

```
## [1]  20 160
```

```r
testset <- testset[,-c(1:7)]
```

Cleaning dataset was performed by removing columns with more than half NA. In addition, training dataset was split into training and validation dataset (6:4 ratio).

```r
CountNA <- sapply(colnames(trainset), function(x) ifelse(sum(is.na(trainset[,x])) > 0.5*nrow(trainset), FALSE, TRUE ))
trainset <- trainset[, CountNA]
dim(trainset)
```

```
## [1] 19622    53
```

```r
inTrain <- createDataPartition(y=trainset$classe, p=0.6, list = FALSE)
trainsetTrain <- trainset[inTrain,]
trainsetTest <- trainset[-inTrain,]
dim(trainsetTrain)
```

```
## [1] 11776    53
```

```r
dim(trainsetTest)
```

```
## [1] 7846   53
```

3 different modeling techniques were used here, including decision tree, random forest, and boosting. Cross validation was used (k fold with K=4) and pca was used for demension reduction. The performance was calculated and shown with confusionMatrix function. It is apparent that random forest has highest accuracy (~0.99) and decision tree is worse (~0.5 accuracy).

```r
trainMC <- trainControl(method = "cv", number = 4, verboseIter=FALSE , preProcOptions="pca", allowParallel=TRUE)

trainTree <- train(classe ~ ., data = trainsetTrain, method = "rpart", trControl= trainMC)
ptrainTree <- predict(trainTree, trainsetTest) 
confusionMatrix(ptrainTree,trainsetTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2041  646  641  563  238
##          B   29  504   38  245  170
##          C  154  368  689  478  388
##          D    0    0    0    0    0
##          E    8    0    0    0  646
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4945          
##                  95% CI : (0.4834, 0.5056)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3388          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9144  0.33202  0.50365   0.0000  0.44799
## Specificity            0.6281  0.92383  0.78574   1.0000  0.99875
## Pos Pred Value         0.4943  0.51116  0.33173      NaN  0.98777
## Neg Pred Value         0.9486  0.85219  0.88230   0.8361  0.88932
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2601  0.06424  0.08782   0.0000  0.08233
## Detection Prevalence   0.5263  0.12567  0.26472   0.0000  0.08335
## Balanced Accuracy      0.7712  0.62792  0.64470   0.5000  0.72337
```

```r
trainRf <- train(classe ~ ., data = trainsetTrain, method = "rf", trControl= trainMC)
ptrainRf <- predict(trainRf, trainsetTest) 
confusionMatrix(ptrainRf,trainsetTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2227   10    0    0    0
##          B    3 1507    9    2    0
##          C    1    1 1348   13    3
##          D    0    0   11 1269    6
##          E    1    0    0    2 1433
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9921          
##                  95% CI : (0.9899, 0.9939)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.99            
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9978   0.9928   0.9854   0.9868   0.9938
## Specificity            0.9982   0.9978   0.9972   0.9974   0.9995
## Pos Pred Value         0.9955   0.9908   0.9868   0.9868   0.9979
## Neg Pred Value         0.9991   0.9983   0.9969   0.9974   0.9986
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2838   0.1921   0.1718   0.1617   0.1826
## Detection Prevalence   0.2851   0.1939   0.1741   0.1639   0.1830
## Balanced Accuracy      0.9980   0.9953   0.9913   0.9921   0.9966
```

```r
trainBoosting <- train(classe ~ ., data = trainsetTrain, method = "gbm", trControl= trainMC)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1291
##      2        1.5232             nan     0.1000    0.0863
##      3        1.4649             nan     0.1000    0.0677
##      4        1.4204             nan     0.1000    0.0497
##      5        1.3855             nan     0.1000    0.0487
##      6        1.3533             nan     0.1000    0.0445
##      7        1.3245             nan     0.1000    0.0339
##      8        1.3013             nan     0.1000    0.0314
##      9        1.2780             nan     0.1000    0.0336
##     10        1.2556             nan     0.1000    0.0286
##     20        1.1011             nan     0.1000    0.0148
##     40        0.9242             nan     0.1000    0.0083
##     60        0.8162             nan     0.1000    0.0068
##     80        0.7389             nan     0.1000    0.0047
##    100        0.6747             nan     0.1000    0.0041
##    120        0.6228             nan     0.1000    0.0023
##    140        0.5783             nan     0.1000    0.0034
##    150        0.5587             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1861
##      2        1.4892             nan     0.1000    0.1287
##      3        1.4051             nan     0.1000    0.1016
##      4        1.3408             nan     0.1000    0.0835
##      5        1.2865             nan     0.1000    0.0698
##      6        1.2413             nan     0.1000    0.0608
##      7        1.2020             nan     0.1000    0.0659
##      8        1.1617             nan     0.1000    0.0462
##      9        1.1297             nan     0.1000    0.0459
##     10        1.0994             nan     0.1000    0.0408
##     20        0.8959             nan     0.1000    0.0219
##     40        0.6836             nan     0.1000    0.0121
##     60        0.5480             nan     0.1000    0.0072
##     80        0.4591             nan     0.1000    0.0039
##    100        0.3937             nan     0.1000    0.0034
##    120        0.3420             nan     0.1000    0.0019
##    140        0.3024             nan     0.1000    0.0034
##    150        0.2839             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2213
##      2        1.4636             nan     0.1000    0.1615
##      3        1.3599             nan     0.1000    0.1241
##      4        1.2809             nan     0.1000    0.1056
##      5        1.2127             nan     0.1000    0.0884
##      6        1.1544             nan     0.1000    0.0684
##      7        1.1099             nan     0.1000    0.0740
##      8        1.0635             nan     0.1000    0.0561
##      9        1.0265             nan     0.1000    0.0514
##     10        0.9929             nan     0.1000    0.0423
##     20        0.7527             nan     0.1000    0.0220
##     40        0.5296             nan     0.1000    0.0110
##     60        0.4069             nan     0.1000    0.0069
##     80        0.3238             nan     0.1000    0.0039
##    100        0.2689             nan     0.1000    0.0028
##    120        0.2256             nan     0.1000    0.0010
##    140        0.1922             nan     0.1000    0.0010
##    150        0.1778             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1289
##      2        1.5242             nan     0.1000    0.0895
##      3        1.4648             nan     0.1000    0.0673
##      4        1.4207             nan     0.1000    0.0568
##      5        1.3847             nan     0.1000    0.0497
##      6        1.3518             nan     0.1000    0.0434
##      7        1.3239             nan     0.1000    0.0358
##      8        1.3009             nan     0.1000    0.0339
##      9        1.2790             nan     0.1000    0.0325
##     10        1.2569             nan     0.1000    0.0303
##     20        1.0990             nan     0.1000    0.0183
##     40        0.9225             nan     0.1000    0.0077
##     60        0.8137             nan     0.1000    0.0076
##     80        0.7334             nan     0.1000    0.0040
##    100        0.6702             nan     0.1000    0.0041
##    120        0.6162             nan     0.1000    0.0028
##    140        0.5711             nan     0.1000    0.0019
##    150        0.5522             nan     0.1000    0.0025
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1801
##      2        1.4877             nan     0.1000    0.1277
##      3        1.4049             nan     0.1000    0.1053
##      4        1.3355             nan     0.1000    0.0823
##      5        1.2811             nan     0.1000    0.0712
##      6        1.2354             nan     0.1000    0.0718
##      7        1.1905             nan     0.1000    0.0569
##      8        1.1530             nan     0.1000    0.0524
##      9        1.1188             nan     0.1000    0.0499
##     10        1.0873             nan     0.1000    0.0357
##     20        0.8884             nan     0.1000    0.0220
##     40        0.6706             nan     0.1000    0.0135
##     60        0.5448             nan     0.1000    0.0049
##     80        0.4618             nan     0.1000    0.0038
##    100        0.3972             nan     0.1000    0.0038
##    120        0.3458             nan     0.1000    0.0029
##    140        0.3008             nan     0.1000    0.0028
##    150        0.2814             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2300
##      2        1.4602             nan     0.1000    0.1578
##      3        1.3575             nan     0.1000    0.1183
##      4        1.2800             nan     0.1000    0.1099
##      5        1.2131             nan     0.1000    0.0917
##      6        1.1553             nan     0.1000    0.0718
##      7        1.1088             nan     0.1000    0.0880
##      8        1.0550             nan     0.1000    0.0554
##      9        1.0180             nan     0.1000    0.0622
##     10        0.9779             nan     0.1000    0.0522
##     20        0.7448             nan     0.1000    0.0248
##     40        0.5230             nan     0.1000    0.0139
##     60        0.3960             nan     0.1000    0.0057
##     80        0.3139             nan     0.1000    0.0031
##    100        0.2579             nan     0.1000    0.0034
##    120        0.2136             nan     0.1000    0.0018
##    140        0.1811             nan     0.1000    0.0015
##    150        0.1666             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1306
##      2        1.5239             nan     0.1000    0.0901
##      3        1.4633             nan     0.1000    0.0668
##      4        1.4180             nan     0.1000    0.0537
##      5        1.3825             nan     0.1000    0.0419
##      6        1.3537             nan     0.1000    0.0446
##      7        1.3245             nan     0.1000    0.0415
##      8        1.2989             nan     0.1000    0.0323
##      9        1.2762             nan     0.1000    0.0305
##     10        1.2563             nan     0.1000    0.0310
##     20        1.0980             nan     0.1000    0.0171
##     40        0.9248             nan     0.1000    0.0095
##     60        0.8153             nan     0.1000    0.0050
##     80        0.7353             nan     0.1000    0.0033
##    100        0.6745             nan     0.1000    0.0033
##    120        0.6227             nan     0.1000    0.0028
##    140        0.5812             nan     0.1000    0.0021
##    150        0.5626             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1846
##      2        1.4867             nan     0.1000    0.1377
##      3        1.3979             nan     0.1000    0.1020
##      4        1.3327             nan     0.1000    0.0864
##      5        1.2770             nan     0.1000    0.0656
##      6        1.2329             nan     0.1000    0.0706
##      7        1.1876             nan     0.1000    0.0639
##      8        1.1467             nan     0.1000    0.0448
##      9        1.1170             nan     0.1000    0.0490
##     10        1.0865             nan     0.1000    0.0466
##     20        0.8882             nan     0.1000    0.0181
##     40        0.6758             nan     0.1000    0.0106
##     60        0.5546             nan     0.1000    0.0061
##     80        0.4626             nan     0.1000    0.0033
##    100        0.3994             nan     0.1000    0.0039
##    120        0.3469             nan     0.1000    0.0019
##    140        0.3037             nan     0.1000    0.0024
##    150        0.2848             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2345
##      2        1.4591             nan     0.1000    0.1717
##      3        1.3501             nan     0.1000    0.1277
##      4        1.2680             nan     0.1000    0.1048
##      5        1.2016             nan     0.1000    0.0826
##      6        1.1481             nan     0.1000    0.0838
##      7        1.0955             nan     0.1000    0.0743
##      8        1.0488             nan     0.1000    0.0614
##      9        1.0099             nan     0.1000    0.0585
##     10        0.9738             nan     0.1000    0.0406
##     20        0.7533             nan     0.1000    0.0215
##     40        0.5271             nan     0.1000    0.0122
##     60        0.4037             nan     0.1000    0.0065
##     80        0.3209             nan     0.1000    0.0028
##    100        0.2647             nan     0.1000    0.0038
##    120        0.2219             nan     0.1000    0.0019
##    140        0.1883             nan     0.1000    0.0008
##    150        0.1742             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1252
##      2        1.5235             nan     0.1000    0.0900
##      3        1.4638             nan     0.1000    0.0681
##      4        1.4199             nan     0.1000    0.0551
##      5        1.3828             nan     0.1000    0.0429
##      6        1.3534             nan     0.1000    0.0407
##      7        1.3250             nan     0.1000    0.0436
##      8        1.2981             nan     0.1000    0.0312
##      9        1.2781             nan     0.1000    0.0319
##     10        1.2556             nan     0.1000    0.0327
##     20        1.0992             nan     0.1000    0.0158
##     40        0.9251             nan     0.1000    0.0075
##     60        0.8138             nan     0.1000    0.0070
##     80        0.7355             nan     0.1000    0.0049
##    100        0.6717             nan     0.1000    0.0043
##    120        0.6208             nan     0.1000    0.0021
##    140        0.5792             nan     0.1000    0.0027
##    150        0.5589             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1879
##      2        1.4877             nan     0.1000    0.1278
##      3        1.4030             nan     0.1000    0.1045
##      4        1.3345             nan     0.1000    0.0826
##      5        1.2800             nan     0.1000    0.0760
##      6        1.2317             nan     0.1000    0.0674
##      7        1.1876             nan     0.1000    0.0591
##      8        1.1495             nan     0.1000    0.0492
##      9        1.1179             nan     0.1000    0.0414
##     10        1.0906             nan     0.1000    0.0400
##     20        0.8888             nan     0.1000    0.0217
##     40        0.6779             nan     0.1000    0.0101
##     60        0.5492             nan     0.1000    0.0061
##     80        0.4620             nan     0.1000    0.0055
##    100        0.3956             nan     0.1000    0.0034
##    120        0.3437             nan     0.1000    0.0023
##    140        0.3008             nan     0.1000    0.0019
##    150        0.2836             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2326
##      2        1.4596             nan     0.1000    0.1627
##      3        1.3523             nan     0.1000    0.1236
##      4        1.2727             nan     0.1000    0.1043
##      5        1.2059             nan     0.1000    0.0867
##      6        1.1508             nan     0.1000    0.0785
##      7        1.1006             nan     0.1000    0.0666
##      8        1.0579             nan     0.1000    0.0525
##      9        1.0232             nan     0.1000    0.0516
##     10        0.9881             nan     0.1000    0.0497
##     20        0.7445             nan     0.1000    0.0238
##     40        0.5288             nan     0.1000    0.0109
##     60        0.4007             nan     0.1000    0.0062
##     80        0.3202             nan     0.1000    0.0036
##    100        0.2600             nan     0.1000    0.0019
##    120        0.2164             nan     0.1000    0.0010
##    140        0.1839             nan     0.1000    0.0007
##    150        0.1708             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2271
##      2        1.4623             nan     0.1000    0.1550
##      3        1.3616             nan     0.1000    0.1256
##      4        1.2808             nan     0.1000    0.1106
##      5        1.2116             nan     0.1000    0.0894
##      6        1.1546             nan     0.1000    0.0721
##      7        1.1088             nan     0.1000    0.0719
##      8        1.0637             nan     0.1000    0.0685
##      9        1.0216             nan     0.1000    0.0614
##     10        0.9840             nan     0.1000    0.0600
##     20        0.7485             nan     0.1000    0.0246
##     40        0.5266             nan     0.1000    0.0102
##     60        0.4029             nan     0.1000    0.0073
##     80        0.3199             nan     0.1000    0.0038
##    100        0.2629             nan     0.1000    0.0033
##    120        0.2210             nan     0.1000    0.0023
##    140        0.1880             nan     0.1000    0.0016
##    150        0.1744             nan     0.1000    0.0015
```

```r
ptrainBoosting <- predict(trainBoosting, trainsetTest)
confusionMatrix(ptrainBoosting,trainsetTest$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2192   42    0    0    1
##          B   27 1430   50    6   22
##          C    8   37 1293   35   17
##          D    4    1   24 1240   20
##          E    1    8    1    5 1382
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9606          
##                  95% CI : (0.9561, 0.9648)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9502          
##  Mcnemar's Test P-Value : 8.905e-08       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9420   0.9452   0.9642   0.9584
## Specificity            0.9923   0.9834   0.9850   0.9925   0.9977
## Pos Pred Value         0.9808   0.9316   0.9302   0.9620   0.9893
## Neg Pred Value         0.9929   0.9861   0.9884   0.9930   0.9907
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2794   0.1823   0.1648   0.1580   0.1761
## Detection Prevalence   0.2849   0.1956   0.1772   0.1643   0.1781
## Balanced Accuracy      0.9872   0.9627   0.9651   0.9784   0.9780
```

Random forest was used here for final prediction on test dataset since it has highest accuracy, verified with validation dataset.

```r
pfinaltest <- predict(trainRf, testset)
pfinaltest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


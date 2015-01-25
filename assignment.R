#Assignment 'Practical Machine Learning'

# packeges that needed for processing
install.packages("caret")
install.packages("randomForest")
install.packages("rpart")
install.packages("ggplot2")
install.packages("rattle")
install.packages("Hmisc")
install.packages("psych")

library(caret) # general purpose machine learning package
library(randomForest);library(rpart) # decision tree algorithms
library(ggplot2) # general purpose plots
library(rattle) # for 'fancyRpartPlot' function
library(Hmisc) # for 'cut2' function
library(psych) for 'descirbe' function

#Load data
pmlTrain <- read.csv("pml-training.csv")
View(pmlTrain)
pmlTest <- read.csv("pml-testing.csv")
View(pmlTest)

#view and eyeball dataset
head(pmlTrain)
summary(pmlTrain)
describe(pmlTrain)

#reduce colums as many colums have a lot of missing values
pmlTrain_red <- pmlTrain[c(2,8:11,37:49,60:68,84:86,102,113:124,140,151:160)]
names(pmlTest)[160] <- names(pmlTrain)[160] # I like to have the same names
pmlTest_red <- pmlTest[c(2,8:11,37:49,60:68,84:86,102,113:124,140,151:160)]

#eyeball and check characteristics of each feature
str(pmlTrain_red) 
summary(pmlTrain_red)
describe(pmlTrain_red)
num_features <- length(names(pmlTrain_red))-1
num_observations <- length(pmlTrain_red$classe)
round(cor(pmlTrain_red[2:num_features]),2) #correlation analysis
for(i in 2:num_features){
  print(names(pmlTrain_red)[i])
  print(qplot(1:num_observations, pmlTrain_red[,i], data=pmlTrain_red, col=classe, ylab=names(pmlTrain_red)[i], xlab="Observations"))
  Sys.sleep(5)
}
#Result - no easy correlation visible - however variance of the variables differ between classes

#Split dataset into traning & test dataset
set.seed(10000)
inTrain <- createDataPartition(y=pmlTrain_red$classe, p=0.7, list=FALSE)
train <- pmlTrain_red[inTrain,]
test <- pmlTrain_red[-inTrain,]
summary(pmlTrain_red$classe)
summary(train$classe)
summary(test$classe)

# train and test

#1: model with rpart (not accurate - model not usable)
set.seed(11111)
modFit_tree <- train(classe ~ .,method="rpart",data=train)
plot(modFit_tree$finalModel, uniform=TRUE, main="Classification Tree")
text(modFit_tree$finalModel, use.n=TRUE, all=TRUE)
fancyRpartPlot(modFit_tree$finalModel)
confusionMatrix(train$classe,predict(modFit_tree,train)) #Result: Accuracy : 0.5, 95% CI : (0.4916, 0.5084)

#2: model with rf - RandomForest (did not terminate; very long running halted - model not usable)
set.seed(11111)
modFit_ft <- train(classe ~ .,method="rf",data=train,prox=TRUE)
confusionMatrix(train$classe,predict(modFit_ft,train))

#3: Model with PCA and Random Forest (did run very long, but yield a very good result)
set.seed(33333)
prComp <- preProcess(train[,c(-1,-54)],method="pca", thresh=0.9); #prComp
trainPCA <- predict(prComp,train[,c(-1,-54)]); #trainPCA
modFitPCA <- train(train$classe ~ .,method="rf", data=trainPCA); #modFitPCA
trainPCA <- predict(prComp,train[,c(-1,-54)]) #trainPCA
confusionMatrix(train$classe,predict(modFitPCA,trainPCA)) # Result: Accuracy : 1, 95% CI : (0.997, 1)
testPCA <- predict(prComp,test[,c(-1,-54)]); #testPCA
confusionMatrix(test$classe,predict(modFitPCA,testPCA)) # Result: Accuracy : 0.9742, 95% CI : (0.9698, 0.9781)

#4: Model with 'glm' (not working as glm can only handle 2-class outcomes - we need 5-classes)
set.seed(44444)
modFit_glm <- train(classe ~ .,data=train, method="glm")
warnings()

#Final results calculation
finalTestPCA <- predict(prComp,pmlTest_red[,c(-1,-54)]); #finalTestPCA 
answers <- predict(modFitPCA,finalTestPCA); answers

#distribute 'answers' to submit to coursera
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)

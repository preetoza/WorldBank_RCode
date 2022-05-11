
#----------------------Level 1 Data Preparation and Model Making-------------------------

#Loading all the library
library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)
library(tidyverse)
library(tidymodels)

#as we donot want our string to be factor or categorical
myData=read.csv("Rice.csv",stringsAsFactors = FALSE)


names(myData)=c("Text","Label")

#we need to convert our label into facto
myData$Label=as.factor(myData$Label)


#-Splitting our Data set--
#we will need to split our data for running the models

# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
indexes <- createDataPartition(myData$Label, times = 1,
                               p = 0.7, list = FALSE)

train <- myData[indexes,]
test <- myData[-indexes,]


#--Data Prepartation--
# Text analytics requires a lot of data exploration, data pre-processing
# and data wrangling. 

# There are many packages in the R ecosystem for performing text
# analytics. One of the newer packages in quanteda. The quanteda
# package has many useful functions for quickly and easily working
# with text data.
library(quanteda)

#when creating our tokens we will make sure to remove number,punctuation,symbols and any hyphone
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

#with the help of quandary we will able to get a better structured data


# Lower case the tokens.
train.tokens <- tokens_tolower(train.tokens)
#this way there won't be any different token created between If vs if or other Capital vs Uncapital


# Use quanteda's built-in stopword list for English.
train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
#this will help us to remove stopwords like "The" "a" or anything else

# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
#this will help us to combine similar words like (cookies,cookie) into 1 token

# Create our first bag-of-words model.

train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)
#dfm will take our pre processed token and create a frequency chart for each word
#restricting the number of tokens using sum of each coloumn

train.tokens.dfm=dfm_trim(train.tokens.dfm,min_termfreq=15)
#to decrease the token size

# Transform to a matrix and inspect.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
#this will help us to see the document frequency matrix


# Setup a the feature data frame with labels.
train.tokens.df <- cbind(Label = train$Label, data.frame(train.tokens.dfm))
#combing our word frequency with our Label

#sometimes the column name need to be cleaned (for instance look at tra)

# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))
#with the help of make name we will be give better names for our coloumn which we will be utalised later

library(doSNOW)

train.tokens.df <- train.tokens.df[!duplicated(as.list(train.tokens.df))]

#-- Creating a TF- IDF Function

# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}

#-- Making N Gram

# Add bigrams to our feature matrix.
train.tokens <- tokens_ngrams(train.tokens, n = 1:2)

# Transform to dfm and then a matrix.
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)

train.tokens.dfm=dfm_trim(train.tokens.dfm,min_termfreq=12)
#to decrease the token size

train.tokens.matrix <- as.matrix(train.tokens.dfm)


# Normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)


# Calculate the IDF vector that we will use for training and test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)


# Calculate TF-IDF for our training corpus 
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, 
                             idf = train.tokens.idf)


# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)


# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))


# Make a clean data frame.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))


# Clean up unused objects in memory.
gc()


#-SVD--

# We'll leverage the irlba package for our singular value 
# decomposition (SVD). The irlba package allows us to specify
# the number of the most important singular vectors we wish to
# calculate and retain for features.


#with SVD we will be able to retain only the useful features
library(irlba)

# Perform SVD. Specifically, reduce dimensional down to 300 columns
# for our latent semantic analysis (LSA)- helps to reduce dimensional and keep imptortant information
train.irlba <- irlba(t(train.tokens.tfidf), nv = 300, maxit = 600)


#----How to prepare our data for new data

# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
sigma.inverse <- 1 / train.irlba$d
u.transpose <- t(train.irlba$u)
document <- train.tokens.tfidf[1,]
document.hat <- sigma.inverse * u.transpose %*% document

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document.hat[1:10]
train.irlba$v[1, 1:10]


# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).We will combine our SVD to our original data
train.svd <- data.frame(Label = train$Label, train.irlba$v)

myData_rec<-recipe(Label~., data = train.svd)%>%
  step_dummy(all_nominal(),-all_outcomes())

# Defininig Random forest model
rf_model<-rand_forest(trees = 50)%>%
  set_engine("ranger",importance = "impurity")%>%
  set_mode("classification")

rf_tree_wflow<-workflow()%>%
  add_recipe(myData_rec)%>%
  add_model(rf_model)

# K fold cross validation
k_fold <- vfold_cv(train.svd,v=10)

# fitting
rf_fit<-rf_tree_wflow%>%
  fit(train.svd)

# extracting accuracy
rf_fit%>%collect_metrics()



#----------------------Level 2 Data Preparation and Model Making-------------
library(readxl)
myData2 <- read_excel("CLustered_PossibleData_Updated-ML.xlsx")
myData2=myData2%>%
  select(product_name,Category)

names(myData2)=c("Text","Label")

myData2$Label=ifelse(myData2$Label=="biscuits","Biscuit","Non Biscuit")

#we need to convert our label into facto
myData2$Label=as.factor(myData2$Label)

#Splitting the Data--
  
  
#we will need to split our data for running the models
  
# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
indexes <- createDataPartition(myData2$Label, times = 1,
                                 p = 0.7, list = FALSE)

train2 <- myData2[indexes,]
test2 <- myData2[-indexes,]

#-We followed the same steps for Data Preparation

library(quanteda)

#when creating our tokens we will make sure to remove number,punctuation,symbols and any hyphone
train2.tokens <- tokens(train2$Text, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)

#with the help of quandary we will able to get a better structured data


# Lower case the tokens.
train2.tokens <- tokens_tolower(train2.tokens)
#this way there won't be any different token created between If vs if or other Capital vs Uncapital


# Use quanteda's built-in stopword list for English.
train2.tokens <- tokens_select(train2.tokens, stopwords(), 
                               selection = "remove")
#this will help us to remove stopwords like "The" "a" or anything else

# Perform stemming on the tokens.
train2.tokens <- tokens_wordstem(train2.tokens, language = "english")
#this will help us to combine similar words like (cookies,cookie) into 1 token

# Create our first bag-of-words model.

train2.tokens.dfm <- dfm(train2.tokens, tolower = FALSE)
#dfm will take our pre processed token and create a frequency chart for each word
#restricting the number of tokens using sum of each coloumn

train2.tokens.dfm=dfm_trim(train.tokens2.dfm,min_termfreq=15)
#to decrease the token size

# Transform to a matrix and inspect.
train2.tokens.matrix <- as.matrix(train2.tokens.dfm)
#this will help us to see the document frequency matrix

# Creating our TF- IDF Function

# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}


# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}

#-Making N Gram

# Add bigrams to our feature matrix.
train2.tokens <- tokens_ngrams(train2.tokens, n = 1:2)

# Transform to dfm and then a matrix.
train2.tokens.dfm <- dfm(train2.tokens, tolower = FALSE)

train2.tokens.dfm=dfm_trim(train2.tokens.dfm,min_termfreq=12)
#to decrease the token size

train2.tokens.matrix <- as.matrix(train2.tokens.dfm)


# Normalize all documents via TF.
train2.tokens.df <- apply(train2.tokens.matrix, 1, term.frequency)


# Calculate the IDF vector that we will use for training and test data!
train2.tokens.idf <- apply(train2.tokens.matrix, 2, inverse.doc.freq)


# Calculate TF-IDF for our training corpus 
train2.tokens.tfidf <-  apply(train2.tokens.df, 2, tf.idf, 
                              idf = train2.tokens.idf)

# Transpose the matrix
train2.tokens.tfidf <- t(train2.tokens.tfidf)

# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train2.tokens.tfidf))
train2.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train2.tokens.tfidf))


# Make a clean data frame.
train2.tokens.tfidf.df <- cbind(Label = train2$Label, data.frame(train2.tokens.tfidf))
names(train2.tokens.tfidf.df) <- make.names(names(train2.tokens.tfidf.df))

#Making SVD-
library(irlba)

# Perform SVD. Specifically, reduce dimensional down to 300 columns
# for our latent semantic analysis (LSA)- helps to reduce dimensional and keep imptortant information
train2.irlba <- irlba(t(train2.tokens.tfidf), nv = 300, maxit = 600)

# As with TF-IDF, we will need to project new data (e.g., the test data)
# into the SVD semantic space. The following code illustrates how to do
# this using a row of the training data that has already been transformed
# by TF-IDF, per the mathematics illustrated in the slides.
sigma2.inverse <- 1 / train2.irlba$d
u2.transpose <- t(train2.irlba$u)
document2 <- train2.tokens.tfidf[1,]
document2.hat <- sigma2.inverse * u2.transpose %*% document2

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document2.hat[1:10]
train2.irlba$v[1, 1:10]


# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).We will combine our SVD to our original data
train2.svd <- data.frame(Label = train2$Label, train2.irlba$v)

myData2_rec<-recipe(Label~., data = train2.svd)%>%
  step_dummy(all_nominal(),-all_outcomes())


rf2_model<-rand_forest(trees = 50)%>%
  set_engine("ranger",importance = "impurity")%>%
  set_mode("classification")

# workflow
rf2_tree_wflow<-workflow()%>%
  add_recipe(myData2_rec)%>%
  add_model(rf2_model)

# K fold cross validation
k_fold <- vfold_cv(train2.svd,v=10)

# fitting
rf2_fit<-rf2_tree_wflow%>%
  fit(train2.svd)




#----------------------Level 3 Data Preparation and Model Making--------
#--Making CLusters
library(readxl)
library(tm)
library(tidymodels)
library(dplyr)
library(stringr)
library(writexl)
library(ggplot2)
library(tidyverse)
library(clustringr)

# File upload
myData <- read_excel("Desktop/MSBA/GSB 503/CLustered_PossibleData-ML.xlsx")
# Made new datafrane idk why
myData1 <- myData

# Corpus treats each row as a single document
corpus <- Corpus(VectorSource(myData1$product_name))
corpus <- tm_map(corpus, removePunctuation)
# Removes stopwords 
corpus <- tm_map(corpus, removeWords,stopwords("english"))
# Converts words into simple form ex: cookies to cookie
corpus <- tm_map(corpus,stemDocument,"english")

# Splits the words from the product names into individual attributes
tdm <- TermDocumentMatrix(corpus, control = list(minWordLength = c(1,Inf)))
tdm

# remove irrelevant words
t <- removeSparseTerms(tdm,sparse = 0.98)


# Inputs terms into a matrix
m <- as.matrix(t)


# Kmeansm1 <- t(m)
set.seed(999)
k <- 5
kc <- kmeans(m1,k)
kc

# Adds clusters to myData1
myData1 <- data.frame(myData1, kc$cluster)
View(myData1)


# Rename clusters 
myData1$kc.cluster <- ifelse(myData1$kc.cluster == 1,"Chocolate",myData1$kc.cluster)
myData1$kc.cluster <- ifelse(myData1$kc.cluster == 2,"Chocolate Chip",myData1$kc.cluster)
myData1$kc.cluster <- ifelse(myData1$kc.cluster == 3,"Creme",myData1$kc.cluster)
myData1$kc.cluster <- ifelse(myData1$kc.cluster == 4,"Butter",myData1$kc.cluster)
myData1$kc.cluster <- ifelse(myData1$kc.cluster == 5,"Sugar",myData1$kc.cluster)
colnames(myData1)[colnames(myData1) == "kc.cluster"] <- "Level3_Clusters"

# Export dataframe to csv
write.csv(myData1,"CLustered_PossibleData_Updated-ML1", row.names = FALSE)

#--Supervised Machine Learning

library(readxl)

myData3 <- read.csv("CLustered_PossibleData_Updated-ML1")

myData3$Level3_Clusters=ifelse(myData3$Level3_Clusters=="Chip","Chocolate Chip",myData3$Level3_Clusters)
myData3$Level3_Clusters=ifelse(myData3$Level3_Clusters=="Butter","Peanut Butter",myData3$Level3_Clusters)

myData3=myData3%>%
  select(product_name,Level3_Clusters)

names(myData3)=c("Text","Label")



#we need to convert our label into facto
myData3$Label=as.factor(myData3$Label)


#--Splitting the data-
# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
indexes <- createDataPartition(myData3$Label, times = 1,
                               p = 0.7, list = FALSE)

train3 <- myData3[indexes,]
test3 <- myData3[-indexes,]



#-Data Preparation

library(quanteda)

#when creating our tokens we will make sure to remove number,punctuation,symbols and any hyphone
train3.tokens <- tokens(train3$Text, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)

#with the help of quandary we will able to get a better structured data

# Lower case the tokens.
train3.tokens <- tokens_tolower(train3.tokens)
#this way there won't be any different token created between If vs if or other Capital vs Uncapital

# Use quanteda's built-in stopword list for English.
train3.tokens <- tokens_select(train3.tokens, stopwords(), 
                               selection = "remove")
#this will help us to remove stopwords like "The" "a" or anything else

# Perform stemming on the tokens.
train3.tokens <- tokens_wordstem(train3.tokens, language = "english")
#this will help us to combine similar words like (cookies,cookie) into 1 token

# Create our first bag-of-words model.

train3.tokens.dfm <- dfm(train3.tokens, tolower = FALSE)
#dfm will take our pre processed token and create a frequency chart for each word
#restricting the number of tokens using sum of each coloumn

train3.tokens.dfm=dfm_trim(train3.tokens.dfm,min_termfreq=15)
#to decrease the token size

# Transform to a matrix and inspect.
train3.tokens.matrix <- as.matrix(train3.tokens.dfm)
#this will help us to see the document frequency matrix

names(myData3)=c("Text","Label")



#we need to convert our label into facto
myData3$Label=as.factor(myData3$Label)



#--------------Splitting the Data-


#we will need to split our data for running the models

# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
indexes <- createDataPartition(myData3$Label, times = 1,
                               p = 0.7, list = FALSE)

train3 <- myData3[indexes,]
test3 <- myData3[-indexes,]



#-----------Data Preparation-

# Text analytics requires a lot of data exploration, data pre-processing
# and data wrangling. 

# There are many packages in the R ecosystem for performing text
# analytics. One of the newer packages in quanteda. The quanteda
# package has many useful functions for quickly and easily working
# with text data.
library(quanteda)

#when creating our tokens we will make sure to remove number,punctuation,symbols and any hyphone
train3.tokens <- tokens(train3$Text, what = "word", 
                        remove_numbers = TRUE, remove_punct = TRUE,
                        remove_symbols = TRUE, remove_hyphens = TRUE)

#with the help of quandary we will able to get a better structured data


# Lower case the tokens.
train3.tokens <- tokens_tolower(train3.tokens)
#this way there won't be any different token created between If vs if or other Capital vs Uncapital


# Use quanteda's built-in stopword list for English.
train3.tokens <- tokens_select(train3.tokens, stopwords(), 
                               selection = "remove")
#this will help us to remove stopwords like "The" "a" or anything else

# Perform stemming on the tokens.
train3.tokens <- tokens_wordstem(train3.tokens, language = "english")
#this will help us to combine similar words like (cookies,cookie) into 1 token

# Create our first bag-of-words model.

train3.tokens.dfm <- dfm(train3.tokens, tolower = FALSE)
#dfm will take our pre processed token and create a frequency chart for each word
#restricting the number of tokens using sum of each coloumn

train3.tokens.dfm=dfm_trim(train3.tokens.dfm,min_termfreq=15)
#to decrease the token size

# Transform to a matrix and inspect.
train3.tokens.matrix <- as.matrix(train3.tokens.dfm)
#this will help us to see the document frequency matrix

#-Making the tf idf function
# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}

# N-grams allow us to augment our document-term frequency matrices

# Add bigrams to our feature matrix.
train3.tokens <- tokens_ngrams(train3.tokens, n = 1:2)
#make bigram and unigram

# Transform to dfm and then a matrix.
train3.tokens.dfm <- dfm(train3.tokens, tolower = FALSE)

train3.tokens.dfm=dfm_trim(train3.tokens.dfm,min_termfreq=12)
#to decrease the token size

train3.tokens.matrix <- as.matrix(train3.tokens.dfm)


# Normalize all documents via TF.
train3.tokens.df <- apply(train3.tokens.matrix, 1, term.frequency)


# Calculate the IDF vector that we will use for training and test data!
train3.tokens.idf <- apply(train3.tokens.matrix, 2, inverse.doc.freq)


# Calculate TF-IDF for our training corpus 
train3.tokens.tfidf <-  apply(train3.tokens.df, 2, tf.idf, 
                              idf = train3.tokens.idf)


# Transpose the matrix
train3.tokens.tfidf <- t(train3.tokens.tfidf)


# Fix incomplete cases
incomplete.cases <- which(!complete.cases(train3.tokens.tfidf))
train3.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train3.tokens.tfidf))


# Make a clean data frame.


train3.tokens.tfidf.df <- cbind(Label = train3$Label, data.frame(train3.tokens.tfidf))
names(train3.tokens.tfidf.df) <- make.names(names(train3.tokens.tfidf.df))


# Clean up unused objects in memory.
gc()

library(irlba)


# Time the code execution
# Perform SVD. Specifically, reduce dimensional down to 300 columns
# for our latent semantic analysis (LSA)- helps to reduce dimensional and keep imptortant information
train3.irlba <- irlba(t(train3.tokens.tfidf), nv = 300, maxit = 600)

#-New Data Preparation-
sigma3.inverse <- 1 / train3.irlba$d
u3.transpose <- t(train3.irlba$u)
document3 <- train3.tokens.tfidf[1,]
document3.hat <- sigma3.inverse * u3.transpose %*% document3

# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)
document3.hat[1:10]


# Create new feature data frame using our document semantic space of 300
# features (i.e., the V matrix from our SVD).We will combine our SVD to our original data
train3.svd <- data.frame(Label = train3$Label, train3.irlba$v)

library(tidymodels)

myData_rec3<-recipe(Label~., data = train3.svd)%>%
  step_dummy(all_nominal(),-all_outcomes())

# Defininig Random forest model
rf_model3<-rand_forest(trees = 50)%>%
  set_engine("ranger",importance = "impurity")%>%
  set_mode("classification")

# workflow
rf_tree_wflow3<-workflow()%>%
  add_recipe(myData_rec3)%>%
  add_model(rf_model3)

# K fold cross validation
k_fold <- vfold_cv(train3.svd,v=10)

# fitting
rf_fit3<-rf_tree_wflow3%>%
  fit(train3.svd)



#----------------------UI Implementation (Server)------------
library(shiny)
library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
library(randomForest)
library(tidyverse)
library(tidymodels)

library(stringr)
# use the below options code if you wish to increase the file input limit, in this example file input limit is increased from 5MB to 9MB

options(shiny.maxRequestSize = 9*1024^2)
set.seed(1)
shinyServer(function(input,output){
  
  # This reactive function will take the inputs from UI.R and use them for read.table() to read the data from the file. It returns the dataset in the form of a dataframe.
  # file$datapath -> gives the path of the file
  data <- reactive({
    file1 <- input$file
    if(is.null(file1)){return()} 
    read.table(file=file1$datapath, sep=input$sep, header = input$header)
    
  })
  
  
  
  # this reactive output contains the summary of the dataset and display the summary in table format
  output$sum <- renderTable({
    if(is.null(data())){return ()}
    test1.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test1.tokens <- tokens_tolower(test1.tokens)
    test1.tokens <- tokens_select(test1.tokens, stopwords(), 
                                  selection = "remove")
    test1.tokens <- tokens_wordstem(test1.tokens, language = "english")
    test1.tokens <- tokens_ngrams(test1.tokens, n = 1:2)
    test1.tokens.dfm <- dfm(test1.tokens, tolower = FALSE)
    test1.tokens.dfm <- dfm_match(test1.tokens.dfm, featnames(train.tokens.dfm))
    test1.tokens.matrix <- as.matrix(test1.tokens.dfm)
    test1.tokens.df <- apply(test1.tokens.matrix, 1, term.frequency)
    test1.tokens.tfidf <-  apply(test1.tokens.df, 2, tf.idf, idf = train.tokens.idf)
    test1.tokens.tfidf <- t(test1.tokens.tfidf)
    test1.tokens.tfidf[is.na(test1.tokens.tfidf)] <- 0.0
    test1.svd.raw <- t(sigma.inverse * u.transpose %*% t(test1.tokens.tfidf))
    test1.svd <- data.frame(test1.svd.raw)
    predicted_Label=predict(rf_fit,test1.svd)
    predicted_prob1=predict(rf_fit,test1.svd,type="prob")
    a31  <- data.frame(round(100*(pmax(predicted_prob1$.pred_Bakery,predicted_prob1$.pred_Rice))),2)
    colnames(a31)=c("Predicted_Prob1","aaa")
    a1=data.frame(data()$Text,data()$Price,predicted_Label)
    colnames(a1)=c("Product_Name","Product_Price","Level_1_Prediction")
    a1=data.frame(a1,a31)
    a1$Final_Prediction_Level_1=str_c(a1$Level_1_Prediction," ( ",a1$Predicted_Prob1,"% match )")
    a1=a1%>%
      select(Product_Name,Product_Price,Final_Prediction_Level_1)
    colnames(a1)=c("Product Name","Product Price","Level 1 Prediction")
    a1
  })
  
  
  # this is for level 2 prediction
  output$sum2 <- renderTable({
    if(is.null(data())){return ()}
    test1.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test1.tokens <- tokens_tolower(test1.tokens)
    test1.tokens <- tokens_select(test1.tokens, stopwords(), 
                                  selection = "remove")
    test1.tokens <- tokens_wordstem(test1.tokens, language = "english")
    test1.tokens <- tokens_ngrams(test1.tokens, n = 1:2)
    test1.tokens.dfm <- dfm(test1.tokens, tolower = FALSE)
    test1.tokens.dfm <- dfm_match(test1.tokens.dfm, featnames(train.tokens.dfm))
    test1.tokens.matrix <- as.matrix(test1.tokens.dfm)
    test1.tokens.df <- apply(test1.tokens.matrix, 1, term.frequency)
    test1.tokens.tfidf <-  apply(test1.tokens.df, 2, tf.idf, idf = train.tokens.idf)
    test1.tokens.tfidf <- t(test1.tokens.tfidf)
    test1.tokens.tfidf[is.na(test1.tokens.tfidf)] <- 0.0
    test1.svd.raw <- t(sigma.inverse * u.transpose %*% t(test1.tokens.tfidf))
    test1.svd <- data.frame(test1.svd.raw)
    predicted_Label=predict(rf_fit,test1.svd)
    
    test2.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test2.tokens <- tokens_tolower(test2.tokens)
    test2.tokens <- tokens_select(test2.tokens, stopwords(), 
                                  selection = "remove")
    test2.tokens <- tokens_wordstem(test2.tokens, language = "english")
    test2.tokens <- tokens_ngrams(test2.tokens, n = 1:2)
    test2.tokens.dfm <- dfm(test2.tokens, tolower = FALSE)
    test2.tokens.dfm <- dfm_match(test2.tokens.dfm, featnames(train2.tokens.dfm))
    test2.tokens.matrix <- as.matrix(test2.tokens.dfm)
    test2.tokens.df <- apply(test2.tokens.matrix, 1, term.frequency)
    test2.tokens.tfidf <-  apply(test2.tokens.df, 2, tf.idf, idf = train2.tokens.idf)
    test2.tokens.tfidf <- t(test2.tokens.tfidf)
    test2.tokens.tfidf[is.na(test2.tokens.tfidf)] <- 0.0
    test2.svd.raw <- t(sigma2.inverse * u2.transpose %*% t(test2.tokens.tfidf))
    test2.svd <- data.frame(test2.svd.raw)
    predicted2_Label=predict(rf2_fit,test2.svd)
    b=data.frame(data()$Text,data()$Price,predicted_Label,predicted2_Label)
    colnames(b)=c("Product_Name","Product_Price","Level_1_Prediction","Level_2_Prediction")
    predicted_prob1=predict(rf2_fit,test2.svd,type = "prob")
    colnames(predicted_prob1)=c("prob1","prob2")
    a31  <- data.frame(round(100*(pmax(predicted_prob1$prob1,predicted_prob1$prob2))),2)
    colnames(a31)=c("Predicted_Prob1","aaa")
    b=data.frame(b,a31)
    b$Final_Prediction_Level_2=str_c(b$Level_2_Prediction," ( ",b$Predicted_Prob1,"% match )")
    b=b%>%
      select(Product_Name,Product_Price,Level_1_Prediction,Final_Prediction_Level_2)
    for (i in 1:length(b$Level_1_Prediction)){
      if (b$Level_1_Prediction[i]=="Bakery"){
        b
      }else{
        b$Final_Prediction_Level_2[i]="N/A"
      }
    }
    colnames(b)=c("Product Name","Product Price","Level 1 Prediction","Level 2 Prediction")
    b
    
  })
  
  output$sum3 <- renderTable({
    if(is.null(data())){return ()}
    test1.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test1.tokens <- tokens_tolower(test1.tokens)
    test1.tokens <- tokens_select(test1.tokens, stopwords(), 
                                  selection = "remove")
    test1.tokens <- tokens_wordstem(test1.tokens, language = "english")
    test1.tokens <- tokens_ngrams(test1.tokens, n = 1:2)
    test1.tokens.dfm <- dfm(test1.tokens, tolower = FALSE)
    test1.tokens.dfm <- dfm_match(test1.tokens.dfm, featnames(train.tokens.dfm))
    test1.tokens.matrix <- as.matrix(test1.tokens.dfm)
    test1.tokens.df <- apply(test1.tokens.matrix, 1, term.frequency)
    test1.tokens.tfidf <-  apply(test1.tokens.df, 2, tf.idf, idf = train.tokens.idf)
    test1.tokens.tfidf <- t(test1.tokens.tfidf)
    test1.tokens.tfidf[is.na(test1.tokens.tfidf)] <- 0.0
    test1.svd.raw <- t(sigma.inverse * u.transpose %*% t(test1.tokens.tfidf))
    test1.svd <- data.frame(test1.svd.raw)
    predicted_Label=predict(rf_fit,test1.svd)
    
    test2.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test2.tokens <- tokens_tolower(test2.tokens)
    test2.tokens <- tokens_select(test2.tokens, stopwords(), 
                                  selection = "remove")
    test2.tokens <- tokens_wordstem(test2.tokens, language = "english")
    test2.tokens <- tokens_ngrams(test2.tokens, n = 1:2)
    test2.tokens.dfm <- dfm(test2.tokens, tolower = FALSE)
    test2.tokens.dfm <- dfm_match(test2.tokens.dfm, featnames(train2.tokens.dfm))
    test2.tokens.matrix <- as.matrix(test2.tokens.dfm)
    test2.tokens.df <- apply(test2.tokens.matrix, 1, term.frequency)
    test2.tokens.tfidf <-  apply(test2.tokens.df, 2, tf.idf, idf = train2.tokens.idf)
    test2.tokens.tfidf <- t(test2.tokens.tfidf)
    test2.tokens.tfidf[is.na(test2.tokens.tfidf)] <- 0.0
    test2.svd.raw <- t(sigma2.inverse * u2.transpose %*% t(test2.tokens.tfidf))
    test2.svd <- data.frame(test2.svd.raw)
    predicted2_Label=predict(rf2_fit,test2.svd)
    
    
    test3.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test3.tokens <- tokens_tolower(test3.tokens)
    test3.tokens <- tokens_select(test3.tokens, stopwords(), 
                                  selection = "remove")
    test3.tokens <- tokens_wordstem(test3.tokens, language = "english")
    test3.tokens <- tokens_ngrams(test3.tokens, n = 1:2)
    test3.tokens.dfm <- dfm(test3.tokens, tolower = FALSE)
    test3.tokens.dfm <- dfm_match(test3.tokens.dfm, featnames(train3.tokens.dfm))
    test3.tokens.matrix <- as.matrix(test3.tokens.dfm)
    test3.tokens.df <- apply(test3.tokens.matrix, 1, term.frequency)
    test3.tokens.tfidf <-  apply(test3.tokens.df, 2, tf.idf, idf = train3.tokens.idf)
    test3.tokens.tfidf <- t(test3.tokens.tfidf)
    test3.tokens.tfidf[is.na(test3.tokens.tfidf)] <- 0.0
    test3.svd.raw <- t(sigma3.inverse * u3.transpose %*% t(test3.tokens.tfidf))
    test3.svd <- data.frame(test3.svd.raw)
    predicted3_Label=predict(rf_fit3,test3.svd)
    aplabel=data.frame(predicted3_Label)
    aplabel$.pred_class=as.character(aplabel$.pred_class)
    aplabel$.pred_class=ifelse(aplabel$.pred_class=="Butter","Peanut Butter",aplabel$.pred_class)
    aplabel$.pred_class=ifelse(aplabel$.pred_class=="Creme","Creamme",aplabel$.pred_class)
    predicted3_Label_pro=predict(rf_fit3,test3.svd,type="prob")
    a3  <- data.frame(round(100*(pmax(predicted3_Label_pro$.pred_Butter,predicted3_Label_pro$.pred_Chocolate,
                                      predicted3_Label_pro$`.pred_Chocolate Chip`,predicted3_Label_pro$.pred_Creme,
                                      predicted3_Label_pro$.pred_Sugar))),2)
    colnames(a3)=c("Predicted_Prob","aaa")
    b=data.frame(data()$Text,data()$Price,predicted_Label,predicted2_Label,aplabel)
    colnames(b)=c("Product_Name","Product_Price","Predicted_Product_Level_1","Predicted_Product_Level_2","Predicted_Product_Level_3")
    b=data.frame(b,a3)
    b$Final_Prediction_Level_3=str_c(b$Predicted_Product_Level_3," ",b$Predicted_Product_Level_2," ( ",b$Predicted_Prob,"% match )")
    c=b%>%
      select(Product_Name,Product_Price,Predicted_Product_Level_1,Predicted_Product_Level_2,Final_Prediction_Level_3)
    for (i in 1:length(c$Predicted_Product_Level_1)){
      if (c$Predicted_Product_Level_1[i]=="Bakery"){
        c
      }else{
        c$Predicted_Product_Level_2[i]="N/A"
        c$Final_Prediction_Level_3[i]="N/A"
      }
    }
    colnames(c)=c("Product Name","Product Price","Level 1 Prediction","Level 2 Prediction","Level 3 Prediction")
    c
    
  })
  
  # this is for level 3 prediction
  output$sum4 <- renderTable({
    if(is.null(data())){return ()}
    test1.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test1.tokens <- tokens_tolower(test1.tokens)
    test1.tokens <- tokens_select(test1.tokens, stopwords(), 
                                  selection = "remove")
    test1.tokens <- tokens_wordstem(test1.tokens, language = "english")
    test1.tokens <- tokens_ngrams(test1.tokens, n = 1:2)
    test1.tokens.dfm <- dfm(test1.tokens, tolower = FALSE)
    test1.tokens.dfm <- dfm_match(test1.tokens.dfm, featnames(train.tokens.dfm))
    test1.tokens.matrix <- as.matrix(test1.tokens.dfm)
    test1.tokens.df <- apply(test1.tokens.matrix, 1, term.frequency)
    test1.tokens.tfidf <-  apply(test1.tokens.df, 2, tf.idf, idf = train.tokens.idf)
    test1.tokens.tfidf <- t(test1.tokens.tfidf)
    test1.tokens.tfidf[is.na(test1.tokens.tfidf)] <- 0.0
    test1.svd.raw <- t(sigma.inverse * u.transpose %*% t(test1.tokens.tfidf))
    test1.svd <- data.frame(test1.svd.raw)
    predicted_Label=predict(rf_fit,test1.svd)
    
    test2.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test2.tokens <- tokens_tolower(test2.tokens)
    test2.tokens <- tokens_select(test2.tokens, stopwords(), 
                                  selection = "remove")
    test2.tokens <- tokens_wordstem(test2.tokens, language = "english")
    test2.tokens <- tokens_ngrams(test2.tokens, n = 1:2)
    test2.tokens.dfm <- dfm(test2.tokens, tolower = FALSE)
    test2.tokens.dfm <- dfm_match(test2.tokens.dfm, featnames(train2.tokens.dfm))
    test2.tokens.matrix <- as.matrix(test2.tokens.dfm)
    test2.tokens.df <- apply(test2.tokens.matrix, 1, term.frequency)
    test2.tokens.tfidf <-  apply(test2.tokens.df, 2, tf.idf, idf = train2.tokens.idf)
    test2.tokens.tfidf <- t(test2.tokens.tfidf)
    test2.tokens.tfidf[is.na(test2.tokens.tfidf)] <- 0.0
    test2.svd.raw <- t(sigma2.inverse * u2.transpose %*% t(test2.tokens.tfidf))
    test2.svd <- data.frame(test2.svd.raw)
    predicted2_Label=predict(rf2_fit,test2.svd)
    
    
    test3.tokens <- tokens(data()$Text, what = "word", 
                           remove_numbers = TRUE, remove_punct = TRUE,
                           remove_symbols = TRUE, remove_hyphens = TRUE)
    
    test3.tokens <- tokens_tolower(test3.tokens)
    test3.tokens <- tokens_select(test3.tokens, stopwords(), 
                                  selection = "remove")
    test3.tokens <- tokens_wordstem(test3.tokens, language = "english")
    test3.tokens <- tokens_ngrams(test3.tokens, n = 1:2)
    test3.tokens.dfm <- dfm(test3.tokens, tolower = FALSE)
    test3.tokens.dfm <- dfm_match(test3.tokens.dfm, featnames(train3.tokens.dfm))
    test3.tokens.matrix <- as.matrix(test3.tokens.dfm)
    test3.tokens.df <- apply(test3.tokens.matrix, 1, term.frequency)
    test3.tokens.tfidf <-  apply(test3.tokens.df, 2, tf.idf, idf = train3.tokens.idf)
    test3.tokens.tfidf <- t(test3.tokens.tfidf)
    test3.tokens.tfidf[is.na(test3.tokens.tfidf)] <- 0.0
    test3.svd.raw <- t(sigma3.inverse * u3.transpose %*% t(test3.tokens.tfidf))
    test3.svd <- data.frame(test3.svd.raw)
    predicted3_Label=predict(rf_fit3,test3.svd)
    aplabel=data.frame(predicted3_Label)
    aplabel$.pred_class=as.character(aplabel$.pred_class)
    aplabel$.pred_class=ifelse(aplabel$.pred_class=="Butter","Peanut Butter",aplabel$.pred_class)
    aplabel$.pred_class=ifelse(aplabel$.pred_class=="Creme","Creamme",aplabel$.pred_class)
    predicted3_Label_pro=predict(rf_fit3,test3.svd,type="prob")
    a3  <- data.frame(round(100*(pmax(predicted3_Label_pro$.pred_Butter,predicted3_Label_pro$.pred_Chocolate,
                                      predicted3_Label_pro$`.pred_Chocolate Chip`,predicted3_Label_pro$.pred_Creme,
                                      predicted3_Label_pro$.pred_Sugar))),2)
    colnames(a3)=c("Predicted_Prob","aaa")
    b=data.frame(data()$Text,data()$Price,predicted_Label,predicted2_Label,aplabel)
    colnames(b)=c("Product_Name","Product_Price","Predicted_Product_Level_1","Predicted_Product_Level_2","Predicted_Product_Level_3")
    b=data.frame(b,a3)
    b$Final_Prediction_Level_3=str_c(b$Predicted_Prob,"% ",b$Predicted_Product_Level_3," ",b$Predicted_Product_Level_2)
    c=b%>%
      select(Product_Name,Product_Price,Predicted_Product_Level_1,Predicted_Product_Level_2,Final_Prediction_Level_3)
    for (i in 1:length(c$Predicted_Product_Level_1)){
      if (c$Predicted_Product_Level_1[i]=="Bakery"){
        c
      }else{
        c$Predicted_Product_Level_2[i]="N/A"
        c$Final_Prediction_Level_3[i]="N/A"
      }
    }
    c
    
    output$tbOutput <- downloadHandler(
      filename = function() {
        paste("File Output",".csv")
        
      },
      content = function(file){
        write.csv(c,file)
      }
      
      
    )
  })
  
  #view the data set tab
  # This reactive output contains the dataset and display the dataset in table format
  output$table <- renderTable({
    if(is.null(data())){return ()}
    data()
  })
  
  # the following renderUI is used to dynamically generate the tabsets when the file is loaded. Until the file is loaded, app will not show the tabset.
  output$tb <- renderUI({
    if(is.null(data()))
      h6(tags$img(src='44.png', heigth=500, width=500,align="center"))
    else
      tabsetPanel(tabPanel("Data", tableOutput("table")),
                  tabPanel("Level 1 Prediction", tableOutput("sum")),
                  tabPanel("Level 2 Prediction", tableOutput("sum2")),
                  tabPanel("Level 3 Prediction", tableOutput("sum3")),
                  tabPanel("Download", tableOutput("sum4")))
  })
  
})


#----------------------UI Implementation (UI)


#----------------------UI Implementation (UI)--------------

library(shiny)
library(shinyWidgets)
shinyUI(fluidPage(
  theme = bslib::bs_theme(bootswatch = "superhero"),
  h2("Group 2 World Bank Project",align="center"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file","Upload the file"), # fileinput() function is used to get the file upload contorl option
      tags$hr(),
      h5(helpText("Is this the first row the coloumn header? ")),
      checkboxInput(inputId = 'header', label = 'Header', value = FALSE),
      br(),
      radioButtons(inputId = 'sep', label = 'Separator', choices = c(Comma=','), selected = ','),
      downloadButton("tbOutput","Download Output")
    ),
    mainPanel(
      uiOutput("tb")
      
      # use below code if you want the tabset programming in the main panel. If so, then tabset will appear when the app loads for the first time.
      #       tabsetPanel(tabPanel("Summary", verbatimTextOutput("sum")),
      #                   tabPanel("Data", tableOutput("table")))
    )
    
  )
))

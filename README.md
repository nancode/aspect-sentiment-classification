# Aspect-sentiment-classification
By Unaiza Faiz(ufaiz2@uic.edu), Vijaya Nandhini Sivaswamy (vsivas2@uic.edu)

Our goal was to train and build a model to predict positive, negative and neutral polarity labels of an unseen test data based on a given aspect term of an opinionated sentence. Sentiment Analysis refers to the computational process of identifying the emotions or opinions based on a given text.

This project aims at building a supervised learning classification model identifying the polarity of an aspect term provided for a given statement. We used context window to extract words within the window range of the aspect term i.e 4 words to the left and right of the aspect term. Vectorization is performed using TF-IDF scheme and different models are evaluated based on 10-fold cross validation. 

## Data pre-processing techniques:
 Replacing punctuations and special characters
 Stop word removal
 Lemmatization
 Tokenization

## Feature Engineering:
  Using windowing technique
  TF-IDF
 
## Models Attempted:
We implemented 8 models using scikit-learn to classify our training set. The models attempted were: 
1.LinearSVC.
2.Naive Bayes Classifier.
3.Multinomial Naive Bayes Classifier
4.MLP Classifier
5.SGD Classifier
6.Adaboost
7.K-neighbours classifier.
8.Logistic Regression

## Results
LinearSVC was considered as the best model with respect to overall accuracy, precision, recall and F-score for the positive and negative classes, and hence used as the classifier on the held-out set.

## Steps to run the code:
1. Download and unzip the folder
2. Edit the variable file_no in predicttest.py as 1 or 2 (depending on test set) and save 
3. Run the command - python predicttest.py
4. All the employed classifiers are commented in classmode.py

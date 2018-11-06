import pandas
from sklearn.svm import LinearSVC
from sklearn import feature_extraction, model_selection, metrics, naive_bayes
import pickle
import datapreprocessing as dp
from sklearn import neural_network 
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


## reading the training data
colnames = ['example_id', 'text', 'aspect_term', 'term_location', 'classes']
data = pandas.read_csv('./data-1_train.csv', header=0, names=colnames)

text = data.text.tolist()
aspectTerms = data.aspect_term.tolist()

##data preprocessing
cleanedText = dp.cleanStatement(text)
cleanedAspect = dp.cleanStatement(aspectTerms)

#getting the string in context words i.e 3 words to the left and right of the aspect term string
textContext = dp.getContextWindow(cleanedText,cleanedAspect)

#print(data.shape)
#print(data.classes.value_counts())

##Vectorization
vect = feature_extraction.text.TfidfVectorizer(min_df = 0.00125, max_df = 0.7, sublinear_tf=True, use_idf=False, stop_words=u'english', analyzer= 'word', ngram_range=(1,5),lowercase=True)
vect.fit(textContext)
train_dtm = vect.transform(textContext)
train_dtm


#LinearSVC - [1]
X = train_dtm
y = data.classes

classifierLinear=LinearSVC(multi_class='crammer_singer', random_state=0)
preds = model_selection.cross_val_predict(classifierLinear, X, y, cv=10) #10 fold cross validation perhaps ???
print(preds)


##Evaluation Result
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)

print(classifierLinear)
print("\nOverall Acurracy - SVM: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

    

##No idea what this does LOL
classifierLinear.fit(X,y)
pickle.dump(classifierLinear, open('model', 'wb'))

# Naive Bayes Classifier - [2]
clf = naive_bayes.BernoulliNB()
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - BernaulliNB: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

# Multinomial Naive Bayes Classifier - [3]
clf = naive_bayes.MultinomialNB()
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - MultinomialNB: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

# MLP Classifier - [4]
clf = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1)
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - MLP: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")



# SVC Classifier - [5]
clf =  SVC(kernel='rbf', gamma=0.58, C=0.81)
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - SVC: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

# DecisionTreeClassifier - [6]
clf =  DecisionTreeClassifier(random_state=0)
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - DecisionTreeClassifier: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

# RandomForestClassifier - [7]
clf =   RandomForestClassifier(criterion='entropy', n_jobs = 10)
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - RandomForestClassifier: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

#  LogisticRegression - [8]
clf = LogisticRegression()
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy - LogisticRegression: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

#  SGDClassifier - [9]
clf =   SGDClassifier()
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy -SGDClassifier: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

#  GaussianProcessClassifier - requires dense data

#  AdaBoostClassifier - [10]
clf =   AdaBoostClassifier()
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy -AdaBoostClassifier: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")


#  KNeighborsClassifier - [11]
clf =   KNeighborsClassifier(3)
preds = model_selection.cross_val_predict(clf, X, y, cv=10)
accScore = metrics.accuracy_score(y,preds)
labels = [-1, 0, 1]
precision = metrics.precision_score(y,preds,average=None,labels=labels)
recall = metrics.recall_score(y,preds,average=None,labels=labels)
f1Score = metrics.f1_score(y,preds,average=None,labels=labels)
print(clf)
print("\nOverall Acurracy -KNeighborsClassifier: ",accScore,"\n")
for i in range(len(labels)):
    print("Precision of %s class: %f" %(labels[i],precision[i]))
    print("Recall of %s class: %f" %(labels[i],recall[i]))
    print("F1-Score of %s class: %f" %(labels[i],f1Score[i]),"\n")

    
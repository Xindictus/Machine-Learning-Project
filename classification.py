#!/Python27/python
# -*- coding: UTF-8 -*-

from os import path
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import csv
import time
from Plot import PlotRoc
from Accuracy import Accuracy
from sklearn.pipeline import Pipeline

start_time = time.time()

# Get path
d = path.dirname(__file__)

if not path.exists(d+"/Classification_Export"):
    os.makedirs(d+"/Classification_Export")

# Reading train_set.csv and test_set.csv
print "Reading train_set.csv and test_set.csv . . .\n"
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")
dt = pd.read_csv(d+"/test_set.csv", header=0, quoting=3, sep="\t")

# Appending Title to Content (train_set.csv)
print "train_set.csv ==> appending title to content . . ."
for indexTrain, rowTrain in df.iterrows():
    title = " " + rowTrain['Title']
    rowTrain['Content'] += 20 * title

# Appending Title to Content (test_set.csv)
print "test_set.csv ==> appending title to content . . .\n"
for indexTest, rowTest in dt.iterrows():
    title = " " + rowTest['Title']
    rowTest['Content'] += 20 * title

########################################################################
# Prepare csvs
print "Creating a 2-dimensional array for EvaluationMetric_10fold.csv . . ."
w, h = 6, 13
EvalMatrix = [[0 for x_eval in range(w)] for y_eval in range(h)]

EvalMatrix[0][0] = "Statistic Measure"
EvalMatrix[0][1] = "Naive Bayes"
EvalMatrix[0][2] = "KNN"
EvalMatrix[0][3] = "SVM"
EvalMatrix[0][4] = "Random Forest"
EvalMatrix[0][5] = "My Method - SVM"
EvalMatrix[1][0] = "CV Accuracy"

for index in range(2, len(EvalMatrix)):
    EvalMatrix[index][0] = "Accuracy, Fold-{0}".format(index-1)
EvalMatrix[12][0] = "ROC"

print "Open the 2 files . . .\n"
evaluation = open(d+"/Classification_Export/EvaluationMetric_10fold.csv", 'wb')
categories = open(d+"/Classification_Export/testSet_categories.csv", 'wb')

#############################################################################
# PREPROCESSING
print "Preprocessing train_set . . .\n"
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y = le.transform(df["Category"])
X = df['Content']

# Vectorizer
print "Vectorizer . . .\n"
vectorizer = CountVectorizer(stop_words='english', max_features=500)
Xclass = vectorizer.fit_transform(X, Y)

# Transformer
print "Transformer . . .\n"
transformer = TfidfTransformer()
Xclass = transformer.fit_transform(Xclass, Y)

print "Split data in X/Y train and X/Y test . . .\n"
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xclass, Y, test_size=.1, random_state=42)

print "Declaring 10-fold . . .\n"
cv = StratifiedKFold(Y_train, n_folds=10)

###############################################################################
print "Declaring Multinomial Classifier . . .\n"
classifier = MultinomialNB()

print "Finding Accuracy . . ."
accMul = Accuracy(classifier, cv, X_train, Y_train, X_test, Y_test)
accMul = accMul.find_accuracy()

for i in range(0, len(accMul)):
    EvalMatrix[i+1][1] = accMul[i]

print "Plotting ROC for Multinomial Classifier . . .\n"
plotMul = PlotRoc(X, Y, classifier, le.classes_, 'MUL')
EvalMatrix[len(EvalMatrix)-1][1] = plotMul.plot_roc_curve()
###############################################################################
# SVD
print "SVD . . .\n"
svd = TruncatedSVD(n_components=300, random_state=42)
Xclass = svd.fit_transform(Xclass, Y)

print "Split again data in X/Y train and X/Y test . . .\n"
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xclass, Y, test_size=.1, random_state=42)
##############################################################################
print "Declaring K-Nearest Neighbor Classifier . . .\n"
classifier = KNeighborsClassifier()

print "Finding Accuracy . . ."
accKNN = Accuracy(classifier, cv, X_train, Y_train, X_test, Y_test)
accKNN = accKNN.find_accuracy()

for i in range(0, len(accKNN)):
    EvalMatrix[i+1][2] = accKNN[i]

print "Plotting ROC for K-Nearest Neighbor Classifier . . .\n"
plotKNN = PlotRoc(X, Y, classifier, le.classes_, 'KNN')
EvalMatrix[len(EvalMatrix)-1][2] = plotKNN.plot_roc_curve()
##############################################################################
print "Declaring classifier (SVM) . . . "
classifier = svm.SVC(kernel='linear', probability=True)

print "Finding Accuracy . . ."
accSVM = Accuracy(classifier, cv, X_train, Y_train, X_test, Y_test)
accSVM = accSVM.find_accuracy()

for i in range(0, len(accSVM)):
    EvalMatrix[i+1][3] = accSVM[i]

print "Plotting ROC for SVM . . .\n"
plotSVM = PlotRoc(X, Y, classifier, le.classes_, 'SVM')
EvalMatrix[len(EvalMatrix)-1][3] = plotSVM.plot_roc_curve()
###############################################################################
print "Declaring Random Forest Classifier . . .\n"
classifier = RandomForestClassifier()

print "Finding Accuracy . . ."
accRFC = Accuracy(classifier, cv, X_train, Y_train, X_test, Y_test)
accRFC = accRFC.find_accuracy()

for i in range(0, len(accRFC)):
    EvalMatrix[i+1][4] = accRFC[i]

print "Plotting ROC for Random Forest Classifier . . .\n"
plotRFC = PlotRoc(X, Y, classifier, le.classes_, 'RFC')
EvalMatrix[len(EvalMatrix)-1][4] = plotRFC.plot_roc_curve()
###############################################################################
###############################################################################
print "-------------------------------------------\n"
print "----------------CUSTOMIZED-----------------\n"

print "Vectorizer . . ."
vector_test = CountVectorizer(stop_words='english', max_features=2000)
Xcustom = vector_test.fit_transform(X, Y)

print "Transformer . . ."
transf_test = TfidfTransformer()
Xcustom = transf_test.fit_transform(Xcustom, Y)

print "SVD . . ."
svd_test = TruncatedSVD(n_components=1000, random_state=42)
Xcustom = svd_test.fit_transform(Xcustom, Y)

print "Declaring Classifier SVM"
classifier = svm.SVC(kernel='linear', probability=True)

print "Split data in X/Y train and X/Y test . . .\n"
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(Xcustom, Y, test_size=.1, random_state=42)

print "Declaring 10-fold . . .\n"
cv = StratifiedKFold(Y_train, n_folds=10)

print "Finding Accuracy . . ."
accCustom = Accuracy(classifier, cv, X_train, Y_train, X_test, Y_test)
accCustom = accCustom.find_accuracy()

for i in range(0, len(accCustom)):
    EvalMatrix[i+1][5] = accCustom[i]

print "Plotting ROC for Custom Classifier Use . . .\n"
plotCustom = PlotRoc(X, Y, classifier, le.classes_, 'Custom')
EvalMatrix[len(EvalMatrix)-1][5] = plotCustom.plot_roc_curve()
###############################################################################
print "Writing to EvaluationMetric_10fold.csv . . .\n"
wr = csv.writer(evaluation, delimiter=',', quoting=csv.QUOTE_ALL)
for values in EvalMatrix:
    wr.writerow(values)
###############################################################################

print "Preprocessing test_set . . ."
le_test = preprocessing.LabelEncoder()
le_test.fit(df["Category"])
Y_train = le_test.transform(df["Category"])
X_train = df['Content']
X_test = dt['Content']
X_test_id = dt['Id']

vector = CountVectorizer(stop_words='english', max_features=2000)

transf = TfidfTransformer()

svd_cust = TruncatedSVD(n_components=1000, random_state=42)

classifier = svm.SVC(kernel='linear')

print "Pipeline . . ."
pipeline = Pipeline([
    ('vect', vector),
    ('tfidf', transf),
    ('svd', svd_cust),
    ('clf', classifier)
])
print "Pipeline fit . . ."
pipeline.fit(X_train, Y_train)
print "Predicting categories . . .\n"
predicted = pipeline.predict(X_test)
predicted = le_test.inverse_transform(predicted)

print "Writing to testSet_categories.csv . . .\n"
wr = csv.writer(categories, delimiter=',', quoting=csv.QUOTE_ALL)
wr.writerow(["ID", "Predicted Category"])
for i in range(0, len(predicted)):
    values = [X_test_id[i], predicted[i]]
    wr.writerow(values)

print("--- %s seconds ---" % (time.time() - start_time))

#!/Python27/python
# -*- coding: UTF-8 -*-

from __future__ import division

from os import path
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import csv

# Get path
d = path.dirname(__file__)

if not path.exists(d+"/Clustering_Export"):
    os.makedirs(d+"/Clustering_Export")

# Reading train_set.csv
print("Reading train_set.csv . . .\n")
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# PREPROCESSING
print "Processing . . ."
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train = le.transform(df["Category"])
X_train = df['Content']

vectorizer = CountVectorizer(stop_words='english', max_features=500)
transformer = TfidfTransformer()
svd = TruncatedSVD(n_components=100, random_state=42)
clf = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1)

pipeline = Pipeline([
    ('vect', vectorizer),
    ('tfidf', transformer),
    ('svd', svd),
    ('clf', clf)
])

# Simple Pipeline Fit
pipeline.fit(X_train)

predicted = pipeline.predict(X_train)

print "Creating a 2-dimensional array for clustering_KMeans.csv . . ."
w, h = 6, 6
EvalMatrix = [[0 for x_eval in range(w)] for y_eval in range(h)]
EvalMatrix[0][0] = ''
for i in range(len(le.classes_)):
    EvalMatrix[0][i+1] = le.classes_[i]

for i in range(0, 5):
    row = list()
    row.append("Cluster " + repr(i))
    count_0, count_1, count_2, count_3, count_4 = 0, 0, 0, 0, 0
    for j in range(0, len(predicted)):
        if predicted[j] == i:
            if Y_train[j] == 0:
                count_0 += 1
            elif Y_train[j] == 1:
                count_1 += 1
            elif Y_train[j] == 2:
                count_2 += 1
            elif Y_train[j] == 3:
                count_3 += 1
            elif Y_train[j] == 4:
                count_4 += 1
    max_art = count_0 + count_1 + count_2 + count_3 + count_4
    row.extend([
        "%0.4f" % (count_0/max_art),
        "%0.4f" % (count_1/max_art),
        "%0.4f" % (count_2/max_art),
        "%0.4f" % (count_3/max_art),
        "%0.4f" % (count_4/max_art)
    ])
    for w in range(len(row)):
        EvalMatrix[i+1][w] = row[w]

print "Open file . . .\n"
clustering = open(d+"/Clustering_Export/clustering_KMeans.csv", 'wb')

print "Writing to clustering_KMeans.csv . . .\n"
wr = csv.writer(clustering, delimiter=',', quoting=csv.QUOTE_ALL)
for values in EvalMatrix:
    wr.writerow(values)

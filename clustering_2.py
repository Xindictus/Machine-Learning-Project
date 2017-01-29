#!/Python27/python
# -*- coding: UTF-8 -*-

from os import path
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
import logging
from optparse import OptionParser
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline

# Get path
d = path.dirname(__file__)

if not path.exists(d+"/Clustering_Export"):
    os.makedirs(d+"/Clustering_Export")

# Reading train_set.csv
print("Reading train_set.csv . . .\n")
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

# PREPROCESSING
print("Preprocessing train_set . . .\n")
le = preprocessing.LabelEncoder()
le.fit(df["Category"])
Y_train = le.transform(df["Category"])
X_train = df['Content']

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int")
op.add_option("--no-minibatch",
              action="store_false", dest="minibatch", default=True)
op.add_option("--no-idf",
              action="store_false", dest="use_idf", default=True)
op.add_option("--use-hashing",
              action="store_true", default=False)
op.add_option("--n-features", type=int, default=10000)
op.add_option("--verbose",
              action="store_true", dest="verbose", default=False)

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

categories = None

labels = Y_train
true_k = np.unique(labels).shape[0]

print("Extracting features from the training dataset using a sparse vectorizer")
if opts.use_hashing:
    if opts.use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=opts.n_features,
                                   stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       stop_words='english',
                                       non_negative=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
                                 min_df=2, stop_words='english',
                                 use_idf=opts.use_idf)
X = vectorizer.fit_transform(X_train)

if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

km.fit(X)

if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    term_list = list()
    for i in range(true_k):
        print("Cluster %d:" % i)
        for ind in order_centroids[i, :10]:
            # term_list.append(("%s" % (terms[ind])))
            term_list.append(terms[ind])
        term_list = [str(x) for x in term_list]
        print term_list
        print
        term_list = list()

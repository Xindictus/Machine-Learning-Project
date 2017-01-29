#!/Python27/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from os import path
import os


class PlotRoc:

    def __init__(self, X, Y, clf, classes, name):
        self.X = X
        self.Y = Y
        self.clf = clf
        self.classes = classes
        self.name = name

    def plot_roc_curve(self):
        self.Y = label_binarize(self.Y, classes=[0, 1, 2, 3, 4])
        n_classes = self.Y.shape[1]
        if self.name == "Custom":
            vectPlot = CountVectorizer(stop_words='english', max_features=2000)
        else:
            vectPlot = CountVectorizer(stop_words='english', max_features=500)
        self.X = vectPlot.fit_transform(self.X, self.Y)
        # Transformer
        transPlot = TfidfTransformer()
        self.X = transPlot.fit_transform(self.X, self.Y)
        # SVD
        if self.name == "Custom":
            svdPlot = TruncatedSVD(n_components=1000, random_state=42)
            self.X = svdPlot.fit_transform(self.X, self.Y)
        elif self.name != "MUL":
            svdPlot = TruncatedSVD(n_components=300, random_state=42)
            self.X = svdPlot.fit_transform(self.X, self.Y)
        # shuffle and split training and test sets
        X_trainPlot, X_testPlot, Y_trainPlot, Y_testPlot = cross_validation.train_test_split(self.X, self.Y, test_size=.1, random_state=42)
        # Learn to predict each class against the other
        self.clf = OneVsRestClassifier(self.clf)
        y_score = self.clf.fit(X_trainPlot, Y_trainPlot).predict_proba(X_testPlot)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(Y_testPlot[:, i], y_score[:, i])
            fpr[i], tpr[i], _ = roc_curve(Y_testPlot[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        # fpr["micro"], tpr["micro"], _ = roc_curve(Y_testPlot.ravel(), y_score.ravel())
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_testPlot.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ##############################################################################
        # Plot ROC curves for the multiclass problem
        lw = 2
        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     # label='ROC curve of class {0} (area = {1:0.2f})'
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(self.classes[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        d = path.dirname(__file__)
        if not path.exists(d + "/Classification_Export"):
            os.makedirs(d + "/Classification_Export")
        plt.show()
        plt.savefig(d+"/Classification_Export/"+self.name+"-plot.png")
        return "%0.2f" % roc_auc_score(Y_testPlot, y_score)

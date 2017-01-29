#!/Python27/python
# -*- coding: UTF-8 -*-
from sklearn import cross_validation


class Accuracy:

    def __init__(self, clf, cv, X_train, Y_train, X_test, Y_test):
        self.clf = clf
        self.cv = cv
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    def find_accuracy(self):
        acc = list()
        scores = cross_validation.cross_val_score(self.clf, self.X_train, self.Y_train, cv=10, scoring='f1_weighted')
        acc.append(("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
        for i, (train, test) in enumerate(self.cv):
            scores = cross_validation.cross_val_score(self.clf, self.X_train[test], self.Y_train[test], cv=10,
                                                      scoring='f1_weighted')
            acc.append(("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))
        return acc

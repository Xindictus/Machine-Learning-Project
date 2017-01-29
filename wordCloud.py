#!/Python27/python
# -*- coding: UTF-8 -*-

from os import path
import os
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import time

start_time = time.time()
d = path.dirname(__file__)

if not path.exists(d+"/WordCloud_Export"):
    os.makedirs(d+"/WordCloud_Export")

print "Reading articles' csv . . ."
df = pd.read_csv(d+"/train_set.csv", header=0, quoting=3, sep="\t")

categories = [
    "Politics",
    "Film",
    "Football",
    "Business",
    "Technology"
]

print "\nReading stopwords' csv . . ."
s_words = pd.read_csv(d+"/stopwords.csv", header=0, quoting=3, sep=",")

stopwords = STOPWORDS.copy()

print "\nAdding categories to stopwords . . ."
for category in categories:
    stopwords.add(category)
    stopwords.add(category.lower())

print "Adding regular verbs to stopwords . . ."
for regular in s_words['Regular']:
    stopwords.add(regular)

print "Adding irregular verbs to stopwords . . ."
for irregular in s_words['Irregular']:
    stopwords.add(irregular)
stopwords.add("will")

print "Adding numbers to stopwords . . ."
for number in s_words['Number']:
    stopwords.add(number)
stopwords.add("year")

print "\nTraversing article categories . . ."
for category in categories:
    print "\nTraversing " + category + " . . ."
    text = ""
    for index, row in df.iterrows():
        if row['Category'] == category:
            title = " " + row['Title']
            text += row['Content'] + 50 * title + " "
    print "Creating Word Cloud for " + category + " . . ."
    wordcloud = WordCloud(max_font_size=60, stopwords=stopwords, relative_scaling=.5).generate(text)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    plt.savefig(d+"/WordCloud_Export/WordCloud-"+category)

print("--- %s seconds ---" % (time.time() - start_time))

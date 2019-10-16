#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 22:19:24 2018

@author: eduardovoted_classifier
"""
from __future__ import division
from __future__ import unicode_literals

import codecs

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize

from statistics import mode

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

import pickle
import sys

class VoteClassifier(ClassifierI):
    
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        
        return conf
        

with codecs.open("short_reviews/positive.txt", "r",encoding='utf-8', errors='ignore') as fdata:
    short_pos = fdata.read()

with codecs.open("short_reviews/negative.txt", "r",encoding='utf-8', errors='ignore') as fdata:
    short_neg = fdata.read()

documents = []

short_pos_splited = short_pos.split("\n")
short_neg_splited = short_neg.split("\n")

for r in short_pos_splited:
    documents.append( (r, "pos") )

for r in short_neg_splited :
    documents.append( (r, "neg") )
       
all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)    

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words) 
#print all_words.most_common(15)
#print all_words["stupid"]

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features
    
#print find_features(movie_reviews.words('neg/cv000_29416.txt'))

featuresets = [(find_features(rev), category) for (rev, category) in documents]       
random.shuffle(featuresets)
               

# Positive data example
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Negative data example
#training_set = featuresets[100:]
#testing_set = featuresets[:100]


classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Original Naive Bayes Algo accuracy percent",(nltk.classify.accuracy(classifier, testing_set))*100

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print "MNB_classifier  Algo accuracy percent",(nltk.classify.accuracy(MNB_classifier, testing_set))*100

#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print "GaussianNB  Algo accuracy percent",(nltk.classify.accuracy(classifier, testing_set))*100
#
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print "BernoulloNB_classifier  Algo accuracy percent",(nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100

# LogisticRegression, SGDClassifier, SVC, LinearSVC, NuSVC
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print "LogisticRegression_classifier  Algo accuracy percent",(nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print "SGDClassifier_classifier  Algo accuracy percent",(nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100

#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print "SVC_classifier  Algo accuracy percent",(nltk.classify.accuracy(SVC_classifier, testing_set))*100

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print "LinearSVC_classifier  Algo accuracy percent",(nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print "NuSVC_classifier  Algo accuracy percent",(nltk.classify.accuracy(NuSVC_classifier, testing_set))*100

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier, 
                                  SGDClassifier_classifier, 
                                  MNB_classifier,
                                  LogisticRegression_classifier, 
                                  BernoulliNB_classifier)

#print "voted_classifier  Algo accuracy percent",(nltk.classify.accuracy(voted_classifier, testing_set))*100
#print "Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100
#print "Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100
#print "Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100
#print "Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100
#print "Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100

### SAVING PICKLES

#save_classifier = open("naivebayes.pickle", "wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
               
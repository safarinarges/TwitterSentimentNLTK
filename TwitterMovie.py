# -*- coding: utf-8 -*-
"""
@author: Narges
"""

# =============================================================================
# Dont forget to pickle your classifier
# =============================================================================
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier (ClassifierI):
    def __init__ (self, *classifiers):
        self._classfiers = classifiers
        
    def classify (self, features):
        votes= []
        for c in self._classfiers:
            v = c.classify(features)
            votes.append(v)
        return mode (votes)
    def confidence (self, features):
        votes = []
        for c in self._classfiers:
            v = c.classify (features)
            votes.append(v)
            
        choice_value = votes.count(mode(votes))
        conf = choice_value / len (votes)
        return conf
    
short_pos = open ("short_reviews/positive.txt", "r").read()
short_neg = open ("short_reviews/negative.txt", "r").read()

documents = []

for k in short_pos.split('\n'):
    documents.append(k,"pos")
    
for k in short_neg.split('\n'):
    documents.append(k, "neg")


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())
    
    

all_words = nltk.FreqDist (all_words)

word_features = list (all_words.keys())[:5000]
















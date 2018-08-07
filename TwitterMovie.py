# -*- coding: utf-8 -*-
"""
@author: Narges
"""

import nltk
from nltk.classify.scikitlearn import SklearnClassifier
import random
import pickle
import io
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
#short_pos = open ("short_reviews/positive.txt", "r").read()
#short_neg = open ("short_reviews/negative.txt", "r").read()

##io.open(filename, encoding='latin-1')
short_pos = io.open ("short_reviews/positive.txt", encoding='latin-1').read()
short_neg = io.open ("short_reviews/negative.txt", encoding='latin-1').read()

documents = []

for k in short_pos.split('\n'):
    documents.append( (k,"pos") )
    
for k in short_neg.split('\n'):
    documents.append( (k, "neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())
    
    

all_words = nltk.FreqDist (all_words)

word_features = list (all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features [w] = (w in words)
        
    return features

 
features_set = [(find_features (rev), category) for (rev, category) in documents]

random.shuffle(features_set)

training = features_set[:10000]
testing = features_set[10000:]

Naiv_classifier = nltk.NaiveBayesClassifier.train(training)

#Save the classifier
#save_classifier = open("naivebayes.pickle","wb")
#pickle.dump(Naiv_classifier, save_classifier)
#save_classifier.close()

classifier_Naiv = open("naivebayes.pickle","rb")
Naiv_classifier = pickle.load(classifier_Naiv)
classifier_Naiv.close()


print("Naive Bays Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(Naiv_classifier, testing))*100)
Naiv_classifier.show_most_informative_features(7)



MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training)
print("MultinomialNB Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(MultinomialNB_classifier, testing))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training)
print("BernoulliNB Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(BernoulliNB_classifier, testing))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training)
print("LogisticRegression Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(LogisticRegression_classifier, testing))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training)
print("SGDClassifier Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(SGDClassifier_classifier, testing))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training)
print("SVC Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(SVC_classifier, testing))*100)

#LinearSVC_classifier = SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training)
#print("LinearSVC Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(LinearSVC, testing))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training)
print("NuSVC Classifier ALgorithm Accuracy Percentage is:", (nltk.classify.accuracy(NuSVC, testing))*100)


















import pandas as pd
import os
import logging
from gensim.models import Word2Vec
from preprocess import *
import numpy as np
import ngram
from __future__ import print_function
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import gensim
import clean
import ngram
import preprocess
import pandas as pd

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),
        num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print ("Review %d of %d" % (counter, len(reviews)))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs
def getCleanReviews(reviews):
    clean_reviews = []
    for reivew in reviews["Text"]:
            clean_reviews.append(preprocess_transcripts(reviews, 
                                                        token_pattern=token_pattern, 
                                                        exclude_stopword=True, 
                                                        stem=True ))
    
import pandas as pd
train = pd.read_csv('manual_dataset_cleaned.csv', sep=',',  encoding='utf-8')
target = ['High','Medium','Low']
targets_dict = dict(zip(target, range(len(target)))) # create numerical values to class labels
train['target'] = map(lambda x: targets_dict[x], 
     train['Class']) # map the labels to text
train['target'].values
print(train.shape) # (97, 4)
n_train = train.shape[0] # take 0 column
print(n_train)

data = train # save the train list
print(type(data.shape))

test = pd.read_csv('911_cleaned.csv', sep=',', encoding='utf-8')

data = pd.concat((train,test))


print(data.shape)
train = data[~data['target'].isnull()] # take non null values
print (type(train))
print(train.shape)
test = data[data['target'].isnull()] # take null values 
print(test.shape)
test.head()
train.head()
test.shape
train.target
test.target

print(train.Text)
#Unigram
data["unigram"] = data["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
data["bigram"] = \
data["unigram"].map(lambda x : ngram.bigram(x, join_str ))
print(type(data["bigram"]))  

data.size
''' data with 6 columns  [157 rows x 6 columns]'''
num_features = 300
min_word_count = 40      
num_workers = 4
context = 10
downsampling = 1e-3

train.columns
sentences = data.unigram.tolist()
print(type(sentences))
model = Word2Vec(sentences, workers = num_workers,
                     size = num_features, min_count= min_word_count,
                     window=context, sample= downsampling,
                     seed=1)


model.init_sims(replace=True)
model_name = '300 Features'
model.save(model_name)
print("Create average Feature Vecs for training")

test["unigram"] = test["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
test["bigram"] = \
test["unigram"].map(lambda x : ngram.bigram(x, join_str ))
print(type(test["bigram"]))

train["unigram"] = \
train["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
train["bigram"] = \
train["unigram"].map(lambda x : ngram.bigram(x, join_str ))
print(type(test["bigram"]))

train.columns
test.columns
train.shape
test.shape
print(train.bigram)
print(train.unigram)
len(data)
N = int(len(data) * 0.1)
print(N)

#labels = list({a['Class'] for a in data})


y_train, y_test = train.Class, test.Class
    
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
X_train = vectorizer.fit_transform(train.Text)
X_test = vectorizer.fit_transform(test.Text)
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
	print('=' * 50)
	print(name)
	results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty)))

print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("OVR")
from sklearn.linear_model import LogisticRegression

results.append(benchmark(OneVsRestClassifier(LogisticRegression())))


print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()

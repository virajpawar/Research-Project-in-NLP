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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


op = OptionParser()
op.add_option("--clean", 
              action="store_true", dest="cleaning",
              help="Clean data for pre-processing...!!!")
op.add_option("--preprocess", 
			  action="store_true", dest="preprocess",
			  help="Preprocess data for feature extraction...!!")

op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")

op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")

op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')


argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)


label = ['High', 'Medium', 'Low']

if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print("Please Clean and preprocess the data....")


if opts.cleaning:
	clean.load_dataset('manual_dataset.csv')
	clean.load_dataset('google_asr_dataset.csv')
	clean.load_dataset('ibm_asr_dataset.csv')
	clean.load_dataset('ms_asr_dataset.csv')
	print("Cleaning of data done \nPlease preprocess the data")


if opts.preprocess:
	#import pandas as pd
	data_train = clean.pandas_dataframe('manual_dataset_cleaned.csv')	
	data_test = clean.pandas_dataframe('ms_asr_dataset_cleaned.csv')
	data_train["Text_unigram"] = data_train["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
	data_test["Text_unigram"] = data_test["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
	print("Pre-processing Done....")
	print(type(data_train.Text_unigram))


train = pd.read_csv('manual_dataset_cleaned.csv', sep=',',  encoding='utf-8')
target = ['High','Medium','Low']
targets_dict = dict(zip(target, range(len(target)))) # create numerical values to class labels
train['target'] = map(lambda x: targets_dict[x], 
     train['Class']) # map the labels to text
train['target'].values
#print(train.shape) # (97, 4)
n_train = train.shape[0] # take 0 column
#print(n_train)

data = train # save the train list

test = pd.read_csv('ibm_asr_dataset_cleaned.csv', sep=',', encoding='utf-8')

data = pd.concat((train,test))



train = data[~data['target'].isnull()] # take non null values

test = data[data['target'].isnull()] # take null values 

#Unigram
data["unigram"] = data["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
data["bigram"] = data["unigram"].map(lambda x : ngram.bigram(x, join_str ))
#print(type(data["bigram"]))  

#data.size

test["unigram"] = test["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
test["bigram"] = test["unigram"].map(lambda x : ngram.bigram(x, join_str ))
print(type(test["bigram"]))

train["unigram"] = train["Text"].map(lambda x: preprocess.preprocess_transcripts(x))
join_str ="_"
train["bigram"] = train["unigram"].map(lambda x : ngram.bigram(x, join_str ))
#print(type(test["bigram"]))

y_train = train['target'].values
y_test = test['target'].values

#print(y_test)

t0 = time()

if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(train.Text_bigram)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(train.Text)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if opts.use_hashing:
	feature_names = None
else:
	features_names = vectorizer.get_features_name()	


if features_names:
	features_names = np.asarray(features_names)

def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."



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

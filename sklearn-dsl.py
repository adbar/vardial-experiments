#!/usr/bin/python3

### Adapted from http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
### Authors P. Prettenhofer, O. Grisel, M. Blondel, and L. Buitinck

### https://github.com/adbar/vardial-experiments
### Adrien Barbaresi, 2017.
### GNU GPL v3 license



from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
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

from sklearn.datasets import load_files


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 22,
              help="n_features when using the hashing vectorizer.") # was 2 ** 16

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

# print(__doc__)
# op.print_help()
# print()


###############################################################################
# Load some categories from the training set
categories = [ 'bs', 'es-AR', 'es-ES', 'es-PE', 'fa-AF', 'fa-IR', 'fr-CA', 'fr-FR', 'hr', 'id', 'my', 'pt-BR', 'pt-PT', 'sr' ]

print("Loading dataset for categories:")
print(categories if categories else "all")


data_train = load_files(container_path='train/', load_content=True, encoding='utf-8', shuffle=False)

data_test = load_files(container_path='test/', load_content=True, encoding='utf-8', shuffle=False)

print('data loaded')

categories = data_train.target_names    # for case categories == None
print (categories)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

# print (data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
# y_train, y_test = data_train.target, data_test.target
y_train = data_train.target


### SET N-GRAM VALUES HERE
print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(non_negative=True, n_features=opts.n_features, analyzer='char', ngram_range=(2,7), strip_accents=None)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,7), strip_accents=None, lowercase=True)
    X_train = vectorizer.fit_transform(data_train.data)



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

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if feature_names:
    feature_names = np.asarray(feature_names)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Predict
def predict(clf):
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
    print("predict time:  %0.3fs" % test_time)


    ### EXP
    pred_prob = clf.predict_proba(X_test)
    i = 0
    with open('test.pred', 'w', encoding='utf-8') as outputfh:
        for item in pred:
            outputfh.write(str(item) + '\n')
            #  + '\t' + data_test.filenames[i] # str(X_test[i]) + '\t' + 
            i += 1
    #with open('test.prob', 'w', encoding='utf-8') as outputfh:
    #    for elem in pred_prob:
    #        for item in elem:
    #            outputfh.write(str(item) + '\t')
    #        outputfh.write('\n')
    ###


    return train_time, test_time


# Benchmark classifiers
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

    # further metrics
    print("f1-micro:   %0.3f" % metrics.f1_score(y_test, pred, average='micro'))
    print("f1-macro:   %0.3f" % metrics.f1_score(y_test, pred, average='macro'))
    print("f1-weighted:   %0.3f" % metrics.f1_score(y_test, pred, average='weighted'))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time


results = []


# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
predict(MultinomialNB(alpha=.005))


#print('=' * 80)
#print("Naive Bayes 2")
#results.append(benchmark(MultinomialNB(alpha=.01)))


# sys.exit('exiting')





#########################################
#########################################

### OTHER TESTS, not optimized

#########################################


# (KNeighborsClassifier(n_neighbors=10), "kNN"),
# (RandomForestClassifier(n_estimators=200, n_jobs=4), "Random forest"),

for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=100, n_jobs=4), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=100, n_jobs=4), "Passive-Aggressive")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                            dual=False, tol=1e-3)))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty=penalty, n_jobs=4)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet", n_jobs=4)))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))


print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
  ('classification', LinearSVC())
])))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

#plt.figure(figsize=(12, 8))
#plt.title("Score")
#plt.barh(indices, score, .2, label="score", color='r')
#plt.barh(indices + .3, training_time, .2, label="training time", color='g')
#plt.barh(indices + .6, test_time, .2, label="test time", color='b')
#plt.yticks(())
#plt.legend(loc='best')
#plt.subplots_adjust(left=.25)
#plt.subplots_adjust(top=.95)
#plt.subplots_adjust(bottom=.05)

#for i, c in zip(indices, clf_names):
#    plt.text(-.3, i, c)

#plt.show()

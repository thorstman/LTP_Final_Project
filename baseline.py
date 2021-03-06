#!usr/bin/python3
#File: baseline.py
#Author: Thijs Horstman
#Date: 01-06-2017
#Description: Naive Bayes baseline for prediction MBTI dimensions for Twitter users.

#---------------------------
#Imports
#---------------------------

#scikit imports
import numpy as np
import argparse
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import preprocessing
from sklearn.utils import shuffle

from sklearn import datasets
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from nltk.tokenize import TweetTokenizer

from collections import defaultdict
import pickle
import codecs
import os
import json
import glob

#---------------------------
#Functions
#---------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="path to json files containing nl or de tweets", type=str, default='nl_200/users_id')
parser.add_argument('--path_meta', help="path to json file containing meta data", type=str, default='TwiSty-NL.json')
parser.add_argument('--lang', help="FOR PICKLE: set language ('nl', 'en', 'de', 'es')", type=str, default='nl')
parser.add_argument('--size', help="FOR PICKLE: number of json files to use", type=int, default=200)
parser.add_argument('--mbti_index', help="mbti index to use (0, 1, 2, 3)", type=int, default=0)
args = parser.parse_args()


#Set parameters
create_pickle = False

#Set parameters
create_pickle = False #creates pickle after reading all json files if set to True

def read_data():
    """Read tweets and metadata from specified paths. Pre-process and return X and y."""
    #corpus location
    path = args.path
    path_meta = args.path_meta
    #initialize vars
    documents, labels = [], []
    filecount = 0
    tokenizer = TweetTokenizer() #NLTK tokenizer    
    
    #extract mbti and tweets of all json files in dir
    for filename in glob.glob(os.path.join(path, '*.json')):
        filecount += 1
        id = os.path.basename(filename)
        id = id.replace('.json', '')
        print("{0}:\tid {1}".format(filecount, id)) #debug
        
        with open(path_meta, mode='r') as metaset:
            meta = json.load(metaset)
            mbti = meta[str(id)]['mbti']
        
        with open(filename, mode="r") as dataset:
            data = json.load(dataset)
            for key, value in data['tweets'].items():
                tokens = tokenizer.tokenize(value['text'].rstrip())
                #tokens = value['text'].strip().split()\
                
                #pre-processing
                tokens = ['URL' if tok[:7] == 'http://' or tok[:8] == 'https://' else tok for tok in tokens]
                tokens = ['@user' if len(tok)>1 and tok[0] == '@' else tok for tok in tokens]
                    
                documents.append(tokens)
                labels.append(mbti)
    
    assert len(documents) == len(labels)
    print("Length documents: {0}".format(len(documents))) #debug
    print("Lenght labels: {0}\n".format(len(labels))) #debug
    
    #shuffle data
    documents, labels = shuffle(documents, labels, random_state=0)
    
    #if enabled, create pickle for faster loading next time
    if create_pickle:
        with open('X_'+args.lang+'_'+str(args.size)+'.pickle', 'wb') as f:
            pickle.dump(documents, f)
        with open('y_'+args.lang+'_'+str(args.size)+'.pickle', 'wb') as f:
            pickle.dump(labels, f)
    
    #use only an index (0, 1, 2 or 3) instead of total MBTI
    labels = [index[args.mbti_index] for index in labels]
            
    return documents, labels

def read_pickle():
    """Only for special use. Read from pickle file for faster loading."""
    #read documents and labels from earlier created pickle instead of going through dir for all json files
    print("Skipping JSON files, loading stored pickle instead...\n")
    with open('C:/Users/Thijs/Desktop/Twisty/tweets/'+args.lang+'_pickle/X_'+args.lang+'_'+str(args.size)+'.pickle', 'rb') as X:
        documents = pickle.load(X)
    with open('C:/Users/Thijs/Desktop/Twisty/tweets/'+args.lang+'_pickle/y_'+args.lang+'_'+str(args.size)+'.pickle', 'rb') as y:
        labels = pickle.load(y)
    #use only an index (0, 1, 2 or 3) instead of total MBTI
    labels = [index[args.mbti_index] for index in labels]
    assert len(documents) == len(labels)
    print("Length documents: {0}".format(len(documents))) #debug
    print("Lenght labels: {0}\n".format(len(labels))) #debug
    
    #slice
    length = len(documents)
    perc = 1
    documents = documents[:int(round(perc*length, 0))]
    labels = labels[:int(round(perc*length, 0))]
    assert len(documents) == len(labels)
    
    return documents, labels

    
def identity(x):
    """Dummy function that just returns its input"""
    return x


def split_data(X, y):
    """Splits data in train, dev, test."""
    train_end = int(0.60*len(X))
    dev_end = int(0.80*len(X))
    
    X_train, X_dev, X_test = X[:train_end], X[train_end:dev_end], X[dev_end:]
    y_train, y_dev, y_test = y[:train_end], y[train_end:dev_end], y[dev_end:]
    assert(len(X_train) == len(y_train))
    assert(len(X_dev) == len(y_dev))
    assert(len(X_test) == len(y_test))
    
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def main():
    #print parameters
    print("Using path {0}".format(args.path))
    if args.mbti_index == 0: print("Using character index {0} for MBTI (E/I)".format(args.mbti_index))
    if args.mbti_index == 1: print("Using character index {0} for MBTI (S/N)".format(args.mbti_index))
    if args.mbti_index == 2: print("Using character index {0} for MBTI (T/F)".format(args.mbti_index))
    if args.mbti_index == 3: print("Using character index {0} for MBTI (J/P)".format(args.mbti_index))
    
    #read and split data
    print("\nLoading data...")
    #X, y = read_pickle() #enable when reading from pickle
    X, y = read_data()
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y)

    # let's use the TF-IDF vectorizer
    tfidf = True
    
    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor = identity,
                              tokenizer = identity)
    else:
        vec = CountVectorizer(preprocessor = identity,
                              tokenizer = identity)

    # combine the vectorizer with a Naive Bayes classifier
    print("Build model...")
    classifier = Pipeline( [('vec', vec),
                            #('cls', svm.SVC(kernel='linear', C=1.0))] )
                            ('cls', MultinomialNB())]) #NaiveBayes
                            #('cls', DecisionTreeClassifier(max_depth=30, min_samples_split=15, max_leaf_nodes=50))]) #DecisionTree
                            #('cls', KNeighborsClassifier(n_neighbors=7))]) #KNearestNeighbors


    # Trainingdata and corresponding labels are put in the classifier to create a model
    # that is capable of predicting labels for unseen instances
    print("Train model...")
    classifier.fit(X_train, y_train)

    print("Evaluate model...")
    y_guess = classifier.predict(X_test)

    print("\nOverall accuracy:", accuracy_score(y_test, y_guess))
    
    # Print classification report with p, r, f1 per class
    print("\nClassification Report:\n", classification_report(y_test, y_guess))
    
    # Print confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_guess))


 

main()
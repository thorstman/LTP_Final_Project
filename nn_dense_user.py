#!usr/bin/python3
#File: nn_dense_user.py
#Author: Thijs Horstman
#Date: 01-06-2017
#Description: sequential neural network with dense input. User-based approach. Predicts binary MBTI dimension.
#Note: some code adapted from B. Plank

#---------------------------
#Imports
#---------------------------

import numpy as np
np.random.seed(113) #set seed before any keras import
import argparse
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, GlobalAveragePooling1D, SimpleRNN, Dropout
from keras.preprocessing import sequence
from keras.optimizers import Adam, SGD

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.utils import shuffle

from nltk.tokenize import TweetTokenizer

from collections import defaultdict
import pickle
import codecs
import os
import json
import glob
import itertools

#set TensorFlow warning level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--path', help="path to json files containing nl or de tweets", type=str, default='nl_10/users_id')
parser.add_argument('--path_meta', help="path to json file containing meta data", type=str, default='TwiSty-NL.json')
parser.add_argument('--lang', help="FOR PICKLE: set language ('nl', 'en', 'de', 'es')", type=str, default='nl')
parser.add_argument('--size', help="FOR PICKLE: number of json files to use", type=int, default=200)
parser.add_argument('--mbti_index', help="mbti index to use (0, 1, 2, 3)", type=int, default=0)
parser.add_argument('--iters', help="epochs (iterations)", type=int, default=6)
args = parser.parse_args()


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
        documents_temp = []
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
                    
                documents_temp.append(tokens)
        
        documents.append(list(itertools.chain(*documents_temp)))
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

    
def word2index(X_train, X_dev, X_test):
    """Function for mapping words to index."""
    w2i = defaultdict(lambda: len(w2i))
    PAD = w2i["<pad>"] # index 0 is padding
    UNK = w2i["<unk>"] # index 1 is for UNK

    #convert words to indices, taking care of UNKs
    X_train_num = [[w2i[word] for word in sentence] for sentence in X_train]
    w2i = defaultdict(lambda: UNK, w2i) # freeze
    X_dev_num = [[w2i[word] for word in sentence] for sentence in X_dev]
    X_test_num = [[w2i[word] for word in sentence] for sentence in X_test]
    
    return X_train_num, X_dev_num, X_test_num, w2i, PAD


def transform_y(y_train, y_dev, y_test):
    """Transform str y to int y. Show mappings."""
    #transform y_train
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    #print(list(le.classes_)) #list all unique label classes
    y_train_binary = le.transform(y_train)
    #num_classes = len(np.unique(y_train_binary)) # how many labels we have
    #print("# classes:", num_classes)
    #y_train_one_hot = np_utils.to_categorical(y_train, num_classes)
    
    #transform y_dev
    le.fit(y_dev)
    y_dev_binary = le.transform(y_dev)
    
    #transform y_test
    le.fit(y_test)
    y_test_binary = le.transform(y_test)
    
    #show integers and mapped labels with inverse transform
    print("0: {0}, 1: {1}".format(le.inverse_transform([0]), le.inverse_transform([1])))
    
    return y_train_binary, y_dev_binary, y_test_binary


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
    
    #convert words to indices
    X_train_num, X_dev_num, X_test_num, w2i, PAD = word2index(X_train, X_dev, X_test)    
    
    #add paddings to X
    max_sentence_length = max([len(s) for s in X_train] + [len(s) for s in X_dev] + [len(s) for s in X_test])    
    X_train_pad = sequence.pad_sequences(X_train_num, maxlen=max_sentence_length, value=PAD)
    X_dev_pad = sequence.pad_sequences(X_dev_num, maxlen=max_sentence_length, value=PAD)
    X_test_pad = sequence.pad_sequences(X_test_num, maxlen=max_sentence_length,value=PAD)
    
    #transform y
    y_train_binary, y_dev_binary, y_test_binary = transform_y(y_train, y_dev, y_test)
    num_classes = len(np.unique(y_train_binary))
    
    vocab_size = len(w2i)
    embeds_size = 64    
    
    #STATISTICS
    print("\nStatistics:")
    print("Max sentence length:", max_sentence_length) #debug
    print("Vocab size:", vocab_size) #debug
    #print("X_train_pad:\n", X_train_pad[-1]) #debug
    #print("X_test_pad:\n", X_test_pad[-1]) #debug
    
    #number of 0/1 labels in train, test and dev data
    print("# train 0:", np.count_nonzero(y_train_binary == 0))
    print("# train 1:", np.count_nonzero(y_train_binary == 1))
    
    print("# dev 0:", np.count_nonzero(y_dev_binary == 0))
    print("# dev 1:", np.count_nonzero(y_dev_binary == 1))
    
    print("# test 0:", np.count_nonzero(y_test_binary == 0))
    print("# test 1:", np.count_nonzero(y_test_binary == 1))
    
    print("Sum 0: {0} ({1}%)".format(int(np.count_nonzero(y_train_binary == 0)) + int(np.count_nonzero(y_dev_binary == 0)) + int(np.count_nonzero(y_test_binary == 0)),
    round((int(np.count_nonzero(y_train_binary == 0)) + int(np.count_nonzero(y_dev_binary == 0)) + int(np.count_nonzero(y_test_binary == 0)))/len(X)*100, 1)))
    print("Sum 1: {0} ({1}%)".format(int(np.count_nonzero(y_train_binary == 1)) + int(np.count_nonzero(y_dev_binary == 1)) + int(np.count_nonzero(y_test_binary == 1)),
    round((int(np.count_nonzero(y_train_binary == 1)) + int(np.count_nonzero(y_dev_binary == 1)) + int(np.count_nonzero(y_test_binary == 1)))/len(X)*100, 1)))
    
    print("X_train pad shape:", X_train_pad.shape) #debug
    print("y_train_binary shape:", y_train_binary.shape) #debug
    
    print("\nBuild model...")
    model = Sequential()
    model.add(Embedding(vocab_size, embeds_size, input_length=max_sentence_length))
    model.add(GlobalAveragePooling1D())
    #model.add(SimpleRNN(32))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.15))
    model.add(Dense(100))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    print("Train model...")
    opt = Adam(lr=0.005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])    
    model.fit(X_train_pad, y_train_binary, epochs=args.iters, batch_size=500)
    
    print("\nEvaluate model...")
    loss, acc = model.evaluate(X_test_pad, y_test_binary)
    print("\nPredict classes...")
    y_predicted_classes = model.predict_classes(X_test_pad)
    print("\n\nLoss:", loss)
    print("Accuracy:", acc)
    print()
    
    print("-----------")
    print("Other metrics:")
    print("Other metrics:")
    print("-----------")
    
    probs = model.predict(X_test_pad)    
    #print("\nProbabilities (last 10):\n", probs[:10])
    y_predicted = [seq.argmax() for seq in probs]    
    
    print("y's predicted (last 20):\n", y_predicted_classes.flatten()[:20])    
    print("y's devset (last 20):\n", y_test_binary[:20])
    #print(probs2[:20])
    print("\nAccuracy_score:", accuracy_score(y_test_binary, y_predicted_classes))
    
    print()
    print("Classification report:\n", classification_report(y_test_binary, y_predicted_classes))
    
    print()
    print("Confusion matrix:\n", confusion_matrix(y_test_binary, y_predicted_classes))
main()

# coding=utf-8
# load and vectorize the corpus from the disk
import os
import logging
import pickle
import numpy as np
from operator import itemgetter
from bidict import bidict
from sklearn.externals import joblib
from text_seg import tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

corpus = [] # tuple of (text, type_id)
typeid2label = dict() # id: description
targetid2label = None  # target_mapping[id] = original id

X_train = None # the training sparse matrix
y_train = None # the target for training data
vectorizer = TfidfVectorizer(min_df=1, tokenizer=tokenize) # used for convert corpus to matrix
clf = None

def load_corpus():
    global typeid2label
    global corpus
    global targetid2label
    # load corpus from hard disk

    logging.info('Loading the mapping file...')
    # load the typeid2label
    with open('./SogouC.ClassList.txt', 'r', encoding='cp936') as c:
        for line in c:
            _id, _name = line.strip().split()
            typeid2label[_id] = _name

    # build the target-label bidict
    labels = list(typeid2label.values())
    targetid2label = bidict([(i, labels[i]) for i in range(0, len(labels))]) # need to be saved

    # load the corpus
    logging.info('loading the corpus...')
    for dp, dn, fn in os.walk('./ClassFile'):
        for f in fn:
            full_path = os.path.join(dp, f)
            par_name = os.path.basename(os.path.dirname(full_path))
            try:
                with open(full_path, 'r', encoding='cp936') as d:
                    text = d.read().strip()
                    data = (text, targetid2label[:typeid2label[par_name]])
                    corpus.append(data)
            except Exception as e:
                logging.error(e)
                continue

    logging.info('Total corpus #{}'.format(len(corpus)))
    return

def vectorize():
    '''
    vectorize the training data
    '''
    global X_train
    global y_train
    logging.info('building the training set...')
    X_train = vectorizer.fit_transform(list(list(zip(*corpus))[0]))
    y_train = np.array(list(list(zip(*corpus))[1]))
    return

def train():
    '''
    Train the corpus
    MODEL: Multinomial Naive Bayes
    '''
    global clf
    logging.info('training the naive bayesian model...')
    clf = MultinomialNB().fit(X_train, y_train)

    # save the model
    logging.info('saving vectorizer and the model...')
    joblib.dump(X_train, 'X_train.pkl')
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(clf, 'nb_classifier.pkl')
    joblib.dump(targetid2label, 'targetid2label.pkl')
    return

def load_model():
    global X_train
    global y_train
    global vectorizer
    global clf
    global targetid2label
    X_train = joblib.load('X_train.pkl')
    y_train = joblib.load('y_train.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    clf = joblib.load('nb_classifier.pkl')
    targetid2label = joblib.load('targetid2label.pkl')

def get_top_words():
    N = 10
    vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary_.items(), key=itemgetter(1))])

    for i in targetid2label.keys():
        topN = np.argsort(clf.coef_[i])[-N:]
        logging.info('The top {} most informative features for topic {} is {}'.format(
            N, targetid2label[i:], ' '.join(vocabulary[topN])
            ))
    return


def train_model():
    logging.info('Model training...')
    load_corpus()
    vectorize()
    train()
    return

def report():
    '''
    report the result of the model
    '''
    logging.info('reporting the result...')
    get_top_words()
    return()


def test():
    train_model()
    report()
    return

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    test()

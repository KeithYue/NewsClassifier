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

def classify(text):
    '''
    given a document, determine which label it belongs to
    '''
    load_model()
    X_predict = vectorizer.transform([text])
    y_predict = clf.predict(X_predict)
    logging.info('This document belongs to label {}'.format(targetid2label[y_predict[0]]))
    return


def test():
    train_model()
    report()
    return

def main():
    '''
    the entry point of the program
    '''
    text = '''
    哈马斯火箭袭击引战端 以军出兵报复
    此轮巴以大规模冲突的导火索，早在数月前就已埋下。4月23日，阿巴斯领导的法塔赫与哈马斯达成和解。此后，巴以龃龉不断。从年初至3月中旬，哈马斯先后向以境内发射至少60枚火箭（注：参见附表），忍无可忍的以色列遂于6月11日空袭加沙北部，造成多名巴勒斯坦人伤亡，双方报复行动由此螺旋式升级。
    6月12日晚，3名犹太神学院学生在以南部希伯伦附近搭便车时失踪。6月14日，以总理内塔尼亚胡宣布失踪青少年“遭恐怖组织绑架”，当晚，以军战机向加沙多个目标发起空袭。次日晨，以军又大举出动，抓捕了包括哈马斯重要领导人哈桑•优素福在内的约150名巴勒斯坦人，地区局势骤然紧张，哈马斯则继续向以境内发射火箭，巴以新一轮冲突的大幕就此拉开。
    6月30日，以方发现了失踪青少年的遗体。从7月1日开始，以陆海空三军齐上阵，对加沙实施攻击。7月8日，以方发起“防务之刃”军事行动。7月17日晚，以军大批地面部队开进加沙，巴以冲突步入短兵相接的白热化阶段。
    哈马斯殴打加沙平民 强迫充当“人盾”
    以色列发动地面战之际，恰值MH17航班在乌克兰东部坠毁，后者无形中掩护了以军行动。虽然以军在空袭或进攻前会散发传单，或打电话、发短信通知巴平民提前撤离，但据外媒7月25日报道，哈马斯殴打、恐吓那些试图离开战区的居民，迫使他们回到住处充当“人体盾牌”，并将火箭发射器藏在人口稠密的居民区、医院、学校和清真寺里。“敌人希望我们向这些目标开火，并伤害无辜旁观者，从而让我们承受国际压力”，以军发言人德洛尔少校说。
    战火毕竟无情，截至8月2日，加沙已有超过1600名巴勒斯坦人丧生，9000多人受伤，其中70%是平民，以方则有63名士兵、3名平民死亡或失踪，超过160人受伤。虽然埃及已允许巴方伤员前往埃及医院救治，外部援助和医生也可经该国进入加沙，但加沙居民生活仍很艰难，部分城镇断水，每天停电20小时，超过46万人逃离家园，其中约半数暂避在联合国修建的61处庇护所内。
    '''
    classify(text)
    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')
    main()

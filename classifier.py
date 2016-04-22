"""
true pozitive = cate sunt 1 din cate erau de fapt 1
false positive = cate sunt 0 in alea care am zis ca sunt 1
true negative = cate sunt 0 din alea care erau 0
false negative = cate sunt 1 din alea care am zis ca sunt 0

precizie = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

f1 = Mh(precizie, recall) = 2 * precizie * recall / (precizie + recall)
"""

from __future__ import print_function
from collections import defaultdict

import cPickle
import json
import logging

from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


log = logging.getLogger(__name__)

DATASETS = defaultdict(list)
TOPICS = {'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'}
PLACES = {'usa', 'uk', 'canada', 'japan', 'west-germany', 'france', 'brazil', 'australia', 'belgium', 'china'}

CLASSIFIERS = {
    'topics': {},
    'places': {}
}

STATS = {
    'topics': defaultdict(lambda : defaultdict(float)),
    'places': defaultdict(lambda : defaultdict(float)),
}

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_data():
    filepath = 'data/train.json'
    with open(filepath, 'r') as f:
        data = json.loads(f.read())
        for item in data:
            title = item.get('title') or ''
            body = item.get('body') or ''
            dateline = item.get('dateline') or ''
            unknown = item.get('unknown') or ''
            topics = item.get('topics', [])
            places = item.get('places', [])

            if not title and not body:
                continue

            body = u'{}\n{}\n{}\n{}'.format(title, body, dateline, unknown)

            # add to training dataset
            if topics:
                DATASETS['data_topics'].append(body)
                DATASETS['target_topics'].append(topics)
                # TOPICS.update(topics)
            if places:
                DATASETS['data_places'].append(body)
                DATASETS['target_places'].append(places)
                # PLACES.update(places)


def train():
    global CLASSIFIERS

    # try:
    #     with open('bin/train.bin', 'rb') as f:
    #         CLASSIFIERS = cPickle.load(f)
    #         return
    # except:
    #     print ('Loading pickle classifiers failed')

    # train topics
    # topics_score = []
    for i, topic in enumerate(TOPICS):
        print ('Training topics:', i, '/', len(TOPICS))

        train_data = DATASETS['data_topics']
        train_target = []
        for index, topics in enumerate(DATASETS['target_topics']):
            train_target.append(int(topic in topics))

        text_clf = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            # ('tfidf', TfidfTransformer()),
            # ('clf', KNeighborsClassifier()),
            ('clf', SGDClassifier(loss='log', random_state=42)),
        ])

        # scores = cross_validation.cross_val_score(text_clf, train_data, train_target, cv=5, scoring='f1_weighted')
        # topics_score.append(scores.mean())
        # print("[%s] Accuracy: %0.4f (+/- %0.4f)" % (topic, scores.mean(), scores.std() * 2))

        text_clf.fit(train_data, train_target)
        CLASSIFIERS['topics'][topic] = text_clf

    # train places
    # places_score = []
    for i, place in enumerate(PLACES):
        print ('Training places:', i, '/', len(PLACES))

        train_data = DATASETS['data_places']
        train_target = []
        for index, places in enumerate(DATASETS['target_places']):
            train_target.append(int(place in places))

        text_clf = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            # ('tfidf', TfidfTransformer()),
            # ('clf', KNeighborsClassifier()),
            ('clf', SGDClassifier(loss='log', random_state=42)),
        ])

        # scores = cross_validation.cross_val_score(text_clf, train_data, train_target, cv=5, scoring='f1_weighted')
        # places_score.append(scores.mean())
        # print("[%s] Accuracy: %0.4f (+/- %0.4f)" % (place, scores.mean(), scores.std() * 2))

        text_clf.fit(train_data, train_target)
        CLASSIFIERS['places'][place] = text_clf

    # print ('[TOTAL] Topics:', sum(topics_score) / len(topics_score) ,'\t Places:', sum(places_score) / len(places_score))

    # try:
    #     with open('bin/train.bin', 'wb') as f:
    #         cPickle.dump(CLASSIFIERS, f)
    # except:
    #     print ('Dumping pickle classifiers failed')

def test():
    filepath = 'data/test.json'
    with open(filepath, 'r') as f:
        data = json.loads(f.read())

        for item in data:
            title = item.get('title') or ''
            body = item.get('body') or ''
            dateline = item.get('dateline') or ''
            unknown = item.get('unknown') or ''
            topics = item.get('topics', [])
            places = item.get('places', [])
            body = u'{}\n{}\n{}\n{}'.format(title, body, dateline, unknown)

            # predict topics
            predicted_topics = set()
            for topic in TOPICS:
                result = CLASSIFIERS['topics'][topic].predict([body])
                if result[0]:
                    predicted_topics.add(topic)

                # stats
                STATS['topics'][topic]['true_positive'] += 1 if result[0] and topic in topics else 0
                STATS['topics'][topic]['false_positive'] += 1 if result[0] and topic not in topics else 0
                # STATS['topics'][topic]['true_negative'] += 1 if not result[0] and topic not in topics else 0
                STATS['topics'][topic]['false_negative'] += 1 if not result[0] and topic in topics else 0

            # predict places
            predicted_places = set()
            for place in PLACES:
                result = CLASSIFIERS['places'][place].predict([body])
                if result[0]:
                    predicted_places.add(place)

                # stats
                STATS['places'][place]['true_positive'] += 1 if result[0] and place in places else 0
                STATS['places'][place]['false_positive'] += 1 if result[0] and place not in places else 0
                # STATS['places'][place]['true_negative'] += 1 if not result[0] and place not in places else 0
                STATS['places'][place]['false_negative'] += 1 if not result[0] and place in places else 0

            # print ('topics', topics, predicted_topics)
            # print ('places', places, predicted_places, '\n\n')

    print ('\n[STATS] Topics:\n---------------------------')
    topics_score = []
    for topic in TOPICS:
        STATS['topics'][topic]['precision'] = safe(lambda: float(STATS['topics'][topic]['true_positive']) / (STATS['topics'][topic]['true_positive'] + STATS['topics'][topic]['false_positive']))
        STATS['topics'][topic]['recall'] = safe(lambda: float(STATS['topics'][topic]['true_positive']) / (STATS['topics'][topic]['true_positive'] + STATS['topics'][topic]['false_negative']))
        STATS['topics'][topic]['f1'] = safe(lambda: 2.0 * STATS['topics'][topic]['precision'] * STATS['topics'][topic]['recall'] / (STATS['topics'][topic]['precision'] + STATS['topics'][topic]['recall']))
        print (topic, bcolors.OKGREEN, STATS['topics'][topic]['f1'], bcolors.ENDC, STATS['topics'][topic])
        topics_score.append(STATS['topics'][topic]['f1'])
    print ('[TOTAL] Topics:', sum(topics_score) / len(topics_score))

    print ('\n[STATS] Places:\n---------------------------')
    places_score = []
    for place in PLACES:
        STATS['places'][place]['precision'] = safe(lambda: float(STATS['places'][place]['true_positive']) / (STATS['places'][place]['true_positive'] + STATS['places'][place]['false_positive']))
        STATS['places'][place]['recall'] = safe(lambda: float(STATS['places'][place]['true_positive']) / (STATS['places'][place]['true_positive'] + STATS['places'][place]['false_negative']))
        STATS['places'][place]['f1'] = safe(lambda: 2.0 * STATS['places'][place]['precision'] * STATS['places'][place]['recall'] / (STATS['places'][place]['precision'] + STATS['places'][place]['recall']))
        print (place, bcolors.OKGREEN, STATS['places'][place]['f1'], bcolors.ENDC, STATS['places'][place])
        places_score.append(STATS['places'][place]['f1'])
    print ('[TOTAL] Places:', sum(places_score) / len(places_score))

def safe(f):
    try:
        return f()
    except ZeroDivisionError:
        return 0

def main():
    print('Starting reading data...')
    read_data()

    print('Start training...')
    train()

    print('Start Testing...')
    test()

    print('\n\nDone.')


if __name__ == '__main__':
    main()

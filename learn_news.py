"""
Learn 20 news groups, using bag-of-words
(later we should try LSTM...)
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import numpy as np
import shutil
import os
from os import path
from os.path import join
import urllib.request
import hashlib


categories = ['alt.atheism', 'soc.religion.christian']


religion_url = 'https://github.com/marcotcr/lime-experiments/blob/master/religion_dataset.tar.gz?raw=true'


def get_md5(filepath):
    with open(filepath, 'rb') as f:
        dat = f.read()
    return hashlib.md5(dat).hexdigest()


def get_religion():
    file_exists = False
    home_dir = os.environ['HOME']
    limedata_dir = join(home_dir, 'limedata')
    if not path.isdir(limedata_dir):
        os.makedirs(limedata_dir)
    religion_filepath = join(limedata_dir, 'religion_dataset.tar.gz')
    if path.isfile(religion_filepath):
        md5sum = get_md5(religion_filepath)
        if md5sum == '0f12beb283869a09584493ddf93672b6':
            file_exists = True
    if not file_exists:
        with urllib.request.urlopen(religion_url) as response, open(religion_filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print(get_md5(religion_filepath))
    # N = 
    # X = []
    # y = []


if __name__ == '__main__':
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=123)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # model = MultinomialNB()
    model = SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, n_iter=5, random_state=123)
    model.fit(X_train_tfidf, twenty_train.target)
    train_pred = model.predict(X_train_tfidf)
    train_num_right = np.equal(train_pred, twenty_train.target).sum()
    print('train', train_num_right, train_num_right / len(twenty_train.target) * 100)

    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=123)
    X_test_counts = count_vect.transform(twenty_test.data)

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    test_pred = model.predict(X_test_tfidf)
    test_num_right = np.equal(test_pred, twenty_test.target).sum()
    print('test', test_num_right, test_num_right / len(twenty_test.target) * 100)

    # now try religion dataset, from https://github.com/marcotcr/lime-experiments/blob/master/religion_dataset.tar.gz
    # download_religion()

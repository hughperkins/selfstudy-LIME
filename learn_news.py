"""
Learn 20 news groups, using bag-of-words
(later we should try LSTM...)
"""
from sklearn.datasets import fetch_20newsgroups
import tarfile
import sklearn.datasets
from collections import defaultdict
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


def download_religion():
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
        print('downloading religion dataset...')
        with urllib.request.urlopen(religion_url) as response, open(religion_filepath, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            print(get_md5(religion_filepath))
            print('... downloaded religion dataset')
    return religion_filepath


def fetch_religion():
    tar_filepath = download_religion()
    # res = sklearn.datasets.base.Bunch()
    examples = []
    print('loading religion dataset to memory...')
    tar = tarfile.open(tar_filepath)
    # print(tar.getmembers())
    class_name_by_id = ['atheism', 'christianity']
    class_id_by_name = {name: id for id, name in enumerate(class_name_by_id)}
    print('class_id_by_name', class_id_by_name)
    N_per_class = 819
    y = np.zeros((N_per_class * 2), dtype=np.int64)
    n = 0
    count_per_class = defaultdict(int)
    for m in tar.getmembers():
        # print(m)
        # print(dir(m))
        # print(m.name, m.path, m.type)
        if '/' in m.path:
            class_name = m.path.split('/')[0]
            class_id = class_id_by_name[class_name]
            if count_per_class[class_id] >= N_per_class:
                continue
            # if m.path not in ['README.txt', 'atheism', 'christianity']:
            f = tar.extractfile(m)
            try:
                content = f.read()
                content = content.decode('utf-8')
            except:
                # raise Exception('failed for [%s]' % content)
                print('failed to decode to utf-8 => skipping 1 doc')
                continue
            finally:
                f.close()
            examples.append(content)
            y[n] = class_id
            count_per_class[class_id] += 1
            n += 1
    tar.close()
    print('... religion dataset loaded')
    return sklearn.datasets.base.Bunch(data=examples, target=y)


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
    religion_test = fetch_religion()
    religion_X_test_counts = count_vect.transform(religion_test.data)
    religion_X_test_tfidf = tfidf_transformer.transform(religion_X_test_counts)
    religion_test_pred = model.predict(religion_X_test_tfidf)
    religion_test_num_right = np.equal(religion_test_pred, religion_test.target).sum()
    print('religion test', religion_test_num_right, religion_test_num_right / len(religion_test.target) * 100)

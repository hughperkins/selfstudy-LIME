import os
from os.path import join
from os import path
import hashlib
import tarfile
import urllib
import shutil
import numpy as np
from collections import defaultdict
import sklearn.datasets


global_categories = ['atheism', 'religion']
news_categories = ['alt.atheism', 'soc.religion.christian']

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
    examples = []
    print('loading religion dataset to memory...')
    tar = tarfile.open(tar_filepath)
    class_name_by_id = ['atheism', 'christianity']
    class_id_by_name = {name: id for id, name in enumerate(class_name_by_id)}
    print('class_id_by_name', class_id_by_name)
    N_per_class = 819
    y = np.zeros((N_per_class * 2), dtype=np.int64)
    n = 0
    count_per_class = defaultdict(int)
    for m in tar.getmembers():
        if '/' in m.path:
            class_name = m.path.split('/')[0]
            class_id = class_id_by_name[class_name]
            if count_per_class[class_id] >= N_per_class:
                continue
            f = tar.extractfile(m)
            try:
                content = f.read()
                content = content.decode('utf-8')
            except:
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

"""
Utilities for downloading and unpacking the CIFAR-10 dataset, originally published
by Krizhevsky et al. and hosted here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

import os
import sys
import tarfile
from six.moves import urllib
import numpy as np

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\n>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def unpickle(file):
    fo = open(file, 'rb')
    if (sys.version_info >= (3, 0)):
        import pickle
        d = pickle.load(fo, encoding='latin1')
    else:
        import cPickle
        d = cPickle.load(fo)
    fo.close()
    return {'x': d['data'].reshape((10000,3,32,32)), 'y': np.array(d['labels']).astype(np.uint8)}

def load_cifar(data_dir="data/cifar_data/"):
    if not os.path.exists(data_dir):
        print('creating folder', data_dir)
        os.makedirs(data_dir)
    maybe_download_and_extract(data_dir)
    train_data = [unpickle(os.path.join(data_dir,'cifar-10-batches-py','data_batch_' + str(i))) for i in range(1,6)]
    skip_first_500 = [0 for x in range(10)]
    trainx_list = []
    trainy_list = []
    valx_list = []
    valy_list = []
    for row in train_data:
        for dx, dy in zip(row['x'], row['y']):
        # print(d['y'])
            if skip_first_500[dy] < 500:
                valx_list.append(dx)
                valy_list.append(dy)
                skip_first_500[dy] += 1
                continue
            trainx_list.append(dx)
            trainy_list.append(dy)
    trainx = np.array(trainx_list)
    trainy = np.array(trainy_list)
    valx = np.array(valx_list)
    valy = np.array(valy_list)
    
    test_data = unpickle(os.path.join(data_dir,'cifar-10-batches-py','test_batch'))
    testx = test_data['x']
    testy = test_data['y']
    trainx = trainx/255.0
    valx = valx/255.0
    testx = testx/255.0
    print("max: " + str(np.amax(trainx)))
    print("min: " + str(np.amin(trainx)))
    print("max: " + str(np.amax(testx)))
    print("min: " + str(np.amin(testx)))
    print("max: " + str(np.amax(valx)))
    print("min: " + str(np.amin(valx)))
    # (N,3,32,32) -> (N,32,32,3)
    return np.transpose(trainx, (0,2,3,1)), \
    np.transpose(valx, (0,2,3,1)), \
    np.transpose(testx, (0,2,3,1))



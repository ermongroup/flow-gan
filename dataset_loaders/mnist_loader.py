import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def load_mnist(send_labels=False):
    mnist = input_data.read_data_sets("data/mnist_data/", one_hot=False)
    np.random.seed(0)
    
    train_data = mnist.train.images[:,:]
    train_labels = mnist.train.labels

    val_data = mnist.train.images[50000:,:]
    val_labels = mnist.train.labels[50000:]
    
    train_data = mnist.train.images[:50000,:]
    train_labels = mnist.train.labels[:50000]
    train_stats = np.zeros((10,), np.float32)
    for label in train_labels:
      train_stats[label] += 1.0/50000.0

    train_data = train_data.reshape((-1,28,28,1))

    val_data = np.concatenate([val_data, mnist.validation.images[:,:]])
    val_labels = np.concatenate([val_labels, mnist.validation.labels])
    val_data = val_data.reshape((-1,28,28,1))
    
    test_data = mnist.test.images[:,:]
    test_labels = mnist.test.labels
    test_data = test_data.reshape((-1,28,28,1))

    if send_labels:
      return train_data, val_data, test_data, train_labels, val_labels, test_labels
    else:
      return train_data, val_data, test_data, train_stats


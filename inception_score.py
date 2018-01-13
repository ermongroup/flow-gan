# From https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def get_inception_and_mode_score(images, splits=10, sess=None):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100
  preds = []
  n_batches = int(math.ceil(float(len(inps)) / float(bs)))
  for i in range(n_batches):
      inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
      inp = np.concatenate(inp, 0)
      pred = sess.run(softmax, {'ExpandDims:0': inp})
      preds.append(pred)
  preds = np.concatenate(preds, 0)
  scores = []
  scores_mode = []
  for i in range(splits):
    part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
    gen_p = np.expand_dims(np.mean(part, 0), 0)
    kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    train_p = np.ones_like(gen_p) * 0.1

    kl_mode_1 = part * (np.log(part) - np.log(train_p))
    kl_mode_1 = np.mean(np.sum(kl_mode_1, 1))
    kl_mode_2 = gen_p * (np.log(gen_p) - np.log(train_p))
    kl_mode_2 = np.mean(np.sum(kl_mode_2, 1))
    scores_mode.append(np.exp(kl_mode_1 - kl_mode_2))
    scores.append(np.exp(kl))
  return np.mean(scores), np.std(scores), np.mean(scores_mode), np.std(scores_mode)

# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  gpu_options = tf.GPUOptions(allow_growth=True)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options\
      , allow_soft_placement=True)) as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()

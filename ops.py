import math
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
  tf.set_random_seed(0)
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    tf.set_random_seed(0)
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    tf.set_random_seed(0) 
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def Layernorm(name, norm_axes, inputs):
  mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

  n_neurons = inputs.get_shape().as_list()[3]

  offset = tf.get_variable(name+'.offset', n_neurons, initializer=tf.constant_initializer(0.0))
  scale = tf.get_variable(name+'.scale', n_neurons, initializer=tf.constant_initializer(1.0))

  result = (inputs - mean) / tf.sqrt(var + 1e-5)
  result = result * scale + offset
  # result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

  return result


def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d", padding="SAME", biases=True, init_type="normal"):
  tf.set_random_seed(0)
  with tf.variable_scope(name):
    input_dim = input_.get_shape().as_list()[-1]
    fan_in = input_dim * k_h**2
    fan_out = output_dim * k_h**2 / (d_h**2)

    init_f = tf.truncated_normal_initializer(stddev=stddev, seed =0)
    if init_type == "he":
      filters_stdev = np.sqrt(4./(fan_in+fan_out))
      init_f = \
      tf.random_uniform_initializer(-np.sqrt(3)*filters_stdev, np.sqrt(3)*filters_stdev, seed=0)
    elif init_type == "glorot":
      filters_stdev = np.sqrt(2./(fan_in+fan_out))
      init_f = \
      tf.random_uniform_initializer(-np.sqrt(3)*filters_stdev, np.sqrt(3)*filters_stdev, seed=0)

    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
            initializer=init_f)

    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    if biases:
      biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
      conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False,init_type="normal"):
  tf.set_random_seed(0)
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    if init_type == "normal":
      init_f = tf.random_normal_initializer(stddev=stddev, seed =0)
    else:
      st_dev = np.sqrt(2./(shape[1]+output_size))
      init_f = tf.random_uniform_initializer(-np.sqrt(3)*st_dev, np.sqrt(3)*st_dev, seed=0)

    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
           init_f)
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias 
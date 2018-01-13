"""
The core Real-NVP model
"""

import tensorflow as tf
import real_nvp.nn as nn
tf.set_random_seed(0)
layers = []

def construct_model_spec(scale_init=2, no_of_layers=8, add_scaling=True):
  global layers
  num_scales = scale_init
  for scale in range(num_scales-1):    
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.SqueezingLayer(name='Squeeze%d' % scale))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_1' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('channel1', name='Channel%d_2' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.CouplingLayer('channel0', name='Channel%d_3' % scale, 
      num_residual_blocks=no_of_layers, scaling=add_scaling))
    layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

  # # final layer
  scale = num_scales-1
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_1' % scale,
      num_residual_blocks=no_of_layers, scaling=add_scaling))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_2' % scale,
      num_residual_blocks=no_of_layers, scaling=add_scaling))
  layers.append(nn.CouplingLayer('checkerboard0', name='Checkerboard%d_3' % scale,
      num_residual_blocks=no_of_layers, scaling=add_scaling))
  layers.append(nn.CouplingLayer('checkerboard1', name='Checkerboard%d_4' % scale,
      num_residual_blocks=no_of_layers, scaling=add_scaling))
  layers.append(nn.FactorOutLayer(scale, name='FactorOut%d' % scale))

def construct_nice_spec(init_type="uniform", hidden_layers=1000, no_of_layers=1):
  global layers

  layers.append(nn.NICECouplingLayer('checkerboard0', name='Checkerboard_1', seed=0, 
    init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
  layers.append(nn.NICECouplingLayer('checkerboard1', name='Checkerboard_2', seed=1, 
    init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
  layers.append(nn.NICECouplingLayer('checkerboard0', name='Checkerboard_3', seed=2, 
    init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
  layers.append(nn.NICECouplingLayer('checkerboard1', name='Checkerboard_4', seed=3, 
    init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers))
  layers.append(nn.NICEScaling(name='Scaling', seed=4))


# the final dimension of the latent space is recorded here
# so that it can be used for constructing the inverse model
final_latent_dimension = []
def model_spec(x, reuse=True, model_type="nice", train=False, 
  alpha=1e-7, init_type="uniform", hidden_layers=1000, no_of_layers=1, batch_norm_adaptive=0):
  counters = {}
  xs = nn.int_shape(x)
  sum_log_det_jacobians = tf.zeros(xs[0])    

  # corrupt data (Tapani Raiko's dequantization)
  y = x
  
  y = y*255.0
  corruption_level = 1.0
  y = y + corruption_level * tf.random_uniform(xs)
  y = y/(255.0 + corruption_level)

  #model logit instead of the x itself
  jac = 0
  

  y = y*(1-2*alpha) + alpha
  if model_type == "nice":
    jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y)+tf.log(1-2*alpha), [1]) 
  else:
    jac = tf.reduce_sum(-tf.log(y) - tf.log(1-y)+tf.log(1-2*alpha), [1,2,3])
  y = tf.log(y) - tf.log(1-y)
  sum_log_det_jacobians += jac

  if len(layers) == 0:
    if model_type == "nice":
      construct_nice_spec(init_type=init_type, hidden_layers=hidden_layers, no_of_layers=no_of_layers)
    else:
      construct_model_spec(no_of_layers=no_of_layers, add_scaling=(batch_norm_adaptive != 0))
  
  # construct forward pass    
  z = None
  jac = sum_log_det_jacobians
  for layer in layers:
    y,jac,z = layer.forward_and_jacobian(y, jac, z, reuse=reuse, train=train)

  if model_type == "nice":
    z = y
  else:
    z = tf.concat(axis=3, values=[z,y])

  # record dimension of the final variable
  global final_latent_dimension
  final_latent_dimension = nn.int_shape(z)

  return z,jac

def inv_model_spec(y, reuse=False, model_type="nice", train=False, alpha=1e-7):
  # construct inverse pass for sampling
  if model_type == "nice":
    z = y
  else:
    shape = final_latent_dimension
    z = tf.reshape(y, [-1, shape[1], shape[2], shape[3]])
    y = None

  for layer in reversed(layers):
    y,z = layer.backward(y,z, reuse=reuse, train=train)

  # inverse logit
  x = y

  x = tf.sigmoid(y)
  x = (x-alpha)/(1-2*alpha)
  return x
    
  

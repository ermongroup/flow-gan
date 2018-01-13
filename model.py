from __future__ import division
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
from six.moves import xrange
import dataset_loaders.cifar_loader as cifar_data
import dataset_loaders.mnist_loader as mnist_data

import scipy
from ops import *
from utils import *

import real_nvp.model as nvp
import real_nvp.nn as nvp_op
import inception_score

class DCGAN(object):
  def __init__(self, sess, input_height=32, input_width=32,
         batch_size=64, sample_num = 64, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default', checkpoint_dir=None,
         f_div='cross-ent', prior="logistic", min_lr=0.0, lr_decay=1.0,
         model_type="nice", alpha=1e-7, log_dir=None,
         init_type="uniform",reg=0.5, n_critic=1.0, hidden_layers=1000,
         no_of_layers= 8, like_reg=0.1, just_sample=False, batch_norm_adaptive=1):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = batch_size
    
    self.input_height = input_height
    self.input_width = input_width
    self.prior = prior

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    self.lr_decay = lr_decay
    self.min_lr = min_lr
    self.model_type = model_type
    self.log_dir = log_dir
    self.alpha = alpha
    self.init_type = init_type
    self.reg = reg
    self.n_critic = n_critic
    self.hidden_layers = hidden_layers
    self.no_of_layers = no_of_layers
    
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.dataset_name = dataset_name
    self.like_reg = like_reg
    if self.dataset_name != 'mnist':
      self.d_bn3 = batch_norm(name='d_bn3')

    self.checkpoint_dir = checkpoint_dir
    self.f_div = f_div
    
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    self.build_model()

  def build_model(self):
    seed =0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
    self.image_size = np.prod(image_dims)
    self.image_dims = image_dims
    if self.dataset_name == "cifar":
      inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.inputs)
    else:
      inputs = self.inputs

    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [self.batch_size, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    #### f: Image Space to Latent space #########
    self.flow_model = tf.make_template('model', 
      lambda x: nvp.model_spec(x, reuse=False, model_type=self.model_type, train=False, 
        alpha=self.alpha, init_type=self.init_type, hidden_layers=self.hidden_layers,
        no_of_layers=self.no_of_layers, batch_norm_adaptive=batch_norm_adaptive), unique_name_='model')

    #### f: Image Space to Latent space for training #########
    self.trainable_flow_model = tf.make_template('model', 
      lambda x: nvp.model_spec(x, reuse=True, model_type=self.model_type, train=True, 
        alpha=self.alpha, init_type=self.init_type, hidden_layers=self.hidden_layers,
        no_of_layers=self.no_of_layers, batch_norm_adaptive=batch_norm_adaptive), unique_name_='model')

    # ##### f^-1: Latent to image (trainable)#######
    self.flow_inv_model = tf.make_template('model', 
      lambda x: nvp.inv_model_spec(x, reuse=True, model_type=self.model_type,
       train=True,alpha=self.alpha), unique_name_='model')
    # ##### f^-1: Latent to image (not-trainable just for sampling)#######
    self.sampler_function = tf.make_template('model', 
      lambda x: nvp.inv_model_spec(x, reuse=True, model_type=self.model_type, 
        alpha=self.alpha,train=False), unique_name_='model')

    
    self.generator_train_batch = self.flow_inv_model
    
    ############### SET SIZE FOR TEST BATCH DEPENDING ON WHETHER WE USE Linear or Conv arch##########
    if self.model_type == "nice":
      self.log_like_batch = tf.placeholder(\
        tf.float32, [self.batch_size, self.image_size], name='log_like_batch')
    elif self.model_type == "real_nvp":
      self.log_like_batch = tf.placeholder(\
        tf.float32, [self.batch_size] + self.image_dims, name='log_like_batch')
    ###############################################

    gen_para, jac = self.flow_model(self.log_like_batch)
    if self.dataset_name == "mnist":
      self.log_likelihood = nvp_op.log_likelihood(gen_para, jac, self.prior)/(self.batch_size)
    else:
      # to calculate values in bits per dim we need to
      # multiply the density by the width of the 
      # discrete probability area, which is 1/256.0, per dimension.
      # The calculation is performed in the log space.
      self.log_likelihood = nvp_op.log_likelihood(gen_para, jac, self.prior)/(self.batch_size)
      self.log_likelihood = 8. + self.log_likelihood / (np.log(2)*self.image_size)

    self.G_before_postprocessing = self.generator_train_batch(self.z)
    self.sampler_before_postprocessing = self.sampler_function(self.z)

    if self.model_type == "real_nvp":
      ##For data dependent init (not completely implemented)
      self.x_init = tf.placeholder(tf.float32, shape=[self.batch_size] + image_dims)
      # run once for data dependent initialization of parameters
      self.trainable_flow_model(self.x_init)
    
    inputs_tr_flow = inputs
    if self.model_type == "nice":
      split_val = int(self.image_size /2)
      self.permutation = np.arange(self.image_size)
      tmp = self.permutation.copy()
      self.permutation[:split_val] = tmp[::2]
      self.permutation[split_val:] = tmp[1::2]
      self.for_perm = np.identity(self.image_size)
      self.for_perm = tf.constant(self.for_perm[:,self.permutation], tf.float32)
      self.rev_perm = np.identity(self.image_size)
      self.rev_perm = tf.constant(self.rev_perm[:,np.argsort(self.permutation)], tf.float32)
      self.G_before_postprocessing \
      = tf.matmul(self.G_before_postprocessing,self.rev_perm)
      self.sampler_before_postprocessing \
      = tf.clip_by_value(tf.matmul(self.sampler_before_postprocessing, self.rev_perm) , 0., 1.)
      inputs_tr_flow = tf.matmul(tf.reshape(inputs, [self.batch_size, self.image_size]), self.for_perm)

    train_gen_para, train_jac = self.trainable_flow_model(inputs_tr_flow)
    self.train_log_likelihood = nvp_op.log_likelihood(train_gen_para, train_jac, self.prior) / self.batch_size
    
    self.sampler = tf.reshape(self.sampler_before_postprocessing, [self.batch_size] + image_dims)
    self.G = tf.reshape(self.G_before_postprocessing, [self.batch_size] + image_dims)

    inputs = inputs*255.0
    corruption_level = 1.0
    inputs = inputs + corruption_level * tf.random_uniform([self.batch_size] + image_dims)
    inputs = inputs/(255.0 + corruption_level)

    self.D, self.D_logits = self.discriminator(inputs, reuse=False)

    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    ### Vanilla gan loss
    if self.f_div == 'ce':
      self.d_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
      self.d_loss_fake = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
      self.g_loss = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    else:
    ### other gan losses
      if self.f_div == 'hellinger':
        self.d_loss_real = tf.reduce_mean(tf.exp(-self.D_logits))
        self.d_loss_fake = tf.reduce_mean(tf.exp(self.D_logits_) - 2.)
        self.g_loss = tf.reduce_mean(tf.exp(-self.D_logits_))
      elif self.f_div == 'rkl':
        self.d_loss_real = tf.reduce_mean(tf.exp(self.D_logits))
        self.d_loss_fake = tf.reduce_mean(-self.D_logits_ - 1.)
        self.g_loss = -tf.reduce_mean(-self.D_logits_ - 1.)
      elif self.f_div == 'kl':
        self.d_loss_real = tf.reduce_mean(-self.D_logits)
        self.d_loss_fake = tf.reduce_mean(tf.exp(self.D_logits_ - 1.))
        self.g_loss = tf.reduce_mean(-self.D_logits_)
      elif self.f_div == 'tv':
        self.d_loss_real = tf.reduce_mean(-0.5 * tf.tanh(self.D_logits))
        self.d_loss_fake = tf.reduce_mean(0.5 * tf.tanh(self.D_logits_))
        self.g_loss = tf.reduce_mean(-0.5 * tf.tanh(self.D_logits_))
      elif self.f_div == 'lsgan':
        self.d_loss_real = 0.5 * tf.reduce_mean((self.D_logits-1)**2)
        self.d_loss_fake = 0.5 * tf.reduce_mean(self.D_logits_**2)
        self.g_loss = 0.5 * tf.reduce_mean((self.D_logits_-1)**2)
      elif self.f_div == "wgan":
        self.g_loss = -tf.reduce_mean(self.D_logits_)
        self.d_loss_real = -tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)
        alpha = tf.random_uniform(
            shape=[1, self.batch_size], 
            minval=0.,
            maxval=1.
        )
        fake_data = self.G
        real_data = inputs
        differences = fake_data - real_data
        interpolates = real_data + \
        tf.transpose(alpha*tf.transpose(differences, perm=[1,2,3,0]), [3,0,1,2])
        _, d_inter = self.discriminator(interpolates, reuse=True) 
        gradients = tf.gradients(d_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((slopes-1.)**2)
      else:
        print("ERROR: Unrecognized f-divergence...exiting")
        exit(-1)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    if self.f_div == "wgan":
      self.d_loss = self.d_loss_real + self.d_loss_fake + self.reg * self.gradient_penalty
    else:
      self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if '/d_' in var.name]
    self.g_vars = [var for var in t_vars if '/g_' in var.name]
    print("gen_vars:")
    for var in self.g_vars:
      print(var.name)

    print("disc_vars:")
    for var in self.d_vars:
      print(var.name)
    
    self.saver = tf.train.Saver(max_to_keep=0)

  def evaluate_neg_loglikelihood(self, data, config):
    log_like_batch_idxs = len(data) // config.batch_size
    lli_list = []
    inter_list = []
    for idx in xrange(0, log_like_batch_idxs):
      batch_images = data[idx*config.batch_size:(idx+1)*config.batch_size]
      batch_images = np.cast[np.float32](batch_images)
      
      if self.model_type == "nice":
        batch_images = batch_images[:,self.permutation]

      lli = self.sess.run([self.log_likelihood],
        feed_dict={self.log_like_batch: batch_images})
      
      lli_list.append(lli)

    return np.mean(lli_list)

  def train(self, config):
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    """Train DCGAN"""
    if config.dataset == "mnist":
      data_X, val_data, test_data, train_dist = mnist_data.load_mnist()
    elif config.dataset == "cifar":
      data_X, val_data, test_data = cifar_data.load_cifar()

    if self.model_type == "nice":
      val_data = np.reshape(val_data, (-1,self.image_size))
      test_data = np.reshape(test_data, (-1, self.image_size))

    lr = config.learning_rate
    self.learning_rate = tf.placeholder(tf.float32, [], name='lr')

    d_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
    d_grad = d_optim_.compute_gradients(self.d_loss, var_list=self.d_vars)
    d_grad_mag = tf.global_norm(d_grad)
    d_optim = d_optim_.apply_gradients(d_grad)          

    g_optim_ = tf.train.AdamOptimizer(self.learning_rate, beta1=config.beta1, beta2=0.9)
    if self.n_critic <= 0:
      g_grad = g_optim_.compute_gradients(self.train_log_likelihood\
          , var_list=self.g_vars)
    else:
      if self.like_reg > 0:
        if self.model_type == "real_nvp":
          g_grad_1 = g_optim_.compute_gradients(self.g_loss / self.like_reg, var_list=self.g_vars)
          g_grad_2 = g_optim_.compute_gradients(self.train_log_likelihood, var_list=self.g_vars)
          grads_1, _ = zip(*g_grad_1)
          grads_2, _ = zip(*g_grad_2)
          sum_grad = [g1+g2 for g1, g2 in zip(grads_1, grads_2)]
          g_grad = [pair for pair in zip(sum_grad, [var for grad, var in g_grad_1])]
        else:
          g_grad = g_optim_.compute_gradients(self.g_loss/self.like_reg + self.train_log_likelihood ,var_list=self.g_vars)  
      else:
        g_grad = g_optim_.compute_gradients(self.g_loss, var_list=self.g_vars)

    
    g_grad_mag = tf.global_norm(g_grad)
    g_optim = g_optim_.apply_gradients(g_grad)         

    try: ##for data-dependent init (not implemented)
      if self.model_type == "real_nvp":
        self.sess.run(tf.global_variables_initializer(),
          {self.x_init: data_X[0:config.batch_size]})
      else:
        self.sess.run(tf.global_variables_initializer())
    except:
      if self.model_type == "real_nvp":
        self.sess.run(tf.global_variables_initializer(),
          {self.x_init: data_X[0:config.batch_size]})
      else:
        self.sess.run(tf.global_variables_initializer())

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./"+self.log_dir, self.sess.graph)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    ############## A FIXED BATCH OF Zs FOR GENERATING SAMPLES ######################
    if self.prior == "uniform":
      sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    elif self.prior == "logistic":
      sample_z = np.random.logistic(loc=0., scale=1., size=(self.sample_num , self.z_dim))
    elif self.prior == "gaussian":
      sample_z = np.random.normal(0.0, 1.0, size=(self.sample_num , self.z_dim))
    else:
        print("ERROR: Unrecognized prior...exiting")
        exit(-1)

    ################################ Evaluate initial model lli ########################

    val_nlli = self.evaluate_neg_loglikelihood(val_data, config)
    # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)

    curr_inception_score = self.calculate_inception_and_mode_score()
    print("INITIAL TEST: val neg logli: %.8f,incep score: %.8f" % (val_nlli,\
     curr_inception_score[0]))
    if counter > 1:
      old_data = np.load("./"+config.sample_dir+'/graph_data.npy') 
      self.best_val_nlli = old_data[2]
      self.best_model_counter = old_data[3]
      self.best_model_path = old_data[4]
      self.val_nlli_list = old_data[1]
      self.counter_list = old_data[5]
      self.batch_train_nlli_list = old_data[-4]
      self.inception_list = old_data[-2]
      self.samples_list = old_data[0]
      self.loss_list = old_data[-1]
      manifold_h, manifold_w = old_data[6]
    else:
      self.writer.add_summary(tf.Summary(\
              value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
      # self.writer.add_summary(tf.Summary(\
      #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)

      self.best_val_nlli = val_nlli
      # self.best_model_train_nlli = train_nlli
      self.best_model_counter = counter
      self.best_model_path = self.save(config.checkpoint_dir, counter)
      # self.train_nlli_list = [train_nlli]
      self.val_nlli_list = [val_nlli]
      self.counter_list = [1]
      self.batch_train_nlli_list = []
      self.inception_list = [curr_inception_score]
      self.samples_list = self.sess.run([self.sampler],
              feed_dict={
                  self.z: sample_z,
              }
            )
      sample_inputs = data_X[0:config.batch_size]
      samples = self.samples_list[0]
      manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
      manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
      self.loss_list = self.sess.run(
              [self.d_loss_real, self.d_loss_fake],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              })
    ##################################################################################

    for epoch in xrange(config.epoch):
      np.random.shuffle(data_X)
      batch_idxs = len(data_X) // config.batch_size
      
      for idx in xrange(0, batch_idxs):
        sys.stdout.flush()
        batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        
        if self.prior == "uniform":
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "logistic":
          batch_z = np.random.logistic(loc=0.,scale=1.0,size=[config.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "gaussian":
          batch_z = np.random.normal(0.0, 1.0, size=(config.batch_size , self.z_dim))
        else:
          print("ERROR: Unrecognized prior...exiting")
          exit(-1)

        for r in range(self.n_critic):
          _, d_g_mag, errD_fake, errD_real ,summary_str = self.sess.run([d_optim, d_grad_mag, 
            self.d_loss_fake, self.d_loss_real, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.learning_rate:lr,
            })
        if self.n_critic > 0:
          self.writer.add_summary(summary_str, counter)

        # Update G network
        if self.like_reg > 0 or self.n_critic <= 0:
          _, g_g_mag, errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.learning_rate:lr,
              self.inputs: batch_images,
            })
        else:
          _, g_g_mag ,errG, summary_str = self.sess.run([g_optim, g_grad_mag, self.g_loss, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.learning_rate:lr,
            })
        self.writer.add_summary(summary_str, counter)

        batch_images_nl = batch_images
        if self.model_type == "nice":
          batch_images_nl = np.reshape(batch_images_nl,(self.batch_size, -1))[:,self.permutation]
        b_train_nlli = self.sess.run([self.log_likelihood], feed_dict={
          self.log_like_batch: batch_images_nl,
          })
        b_train_nlli = b_train_nlli[0]

        self.batch_train_nlli_list.append(b_train_nlli)
        if self.n_critic > 0:
          self.loss_list.append([errD_real, errD_fake])
          self.writer.add_summary(tf.Summary(\
          value=[tf.Summary.Value(tag="training loss", simple_value=-(errD_fake+errD_real))]) ,counter)
        self.writer.add_summary(tf.Summary(\
          value=[tf.Summary.Value(tag="Batch train Neg Log-likelihood", simple_value=b_train_nlli)]) ,counter)
        counter += 1


        lr = max(lr * self.lr_decay, self.min_lr)

        if np.mod(counter, 703) == 1: #340
          if self.n_critic > 0:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_grad_mag: %.8f, g_grad_mag: %.8f, lr: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, d_g_mag, g_g_mag, lr))
          else:
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f, g_grad_mag: %.8f, lr: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errG, g_g_mag, lr))
          curr_model_path = self.save(config.checkpoint_dir, counter)

          val_nlli=self.evaluate_neg_loglikelihood(val_data, config)

          # train_nlli = self.evaluate_neg_loglikelihood(train_data, config)
          curr_inception_score = self.calculate_inception_and_mode_score()

          print("[LogLi (%d,%d)]: val neg logli: %.8f, ince: %.8f, train lli: %.8f" % (epoch, idx,val_nlli,\
           curr_inception_score[0], np.mean(self.batch_train_nlli_list[-700:])))

          self.writer.add_summary(tf.Summary(\
                  value=[tf.Summary.Value(tag="Val Neg Log-likelihood", simple_value=val_nlli)]), counter)
          # self.writer.add_summary(tf.Summary(\
          #         value=[tf.Summary.Value(tag="Train Neg Log-likelihood", simple_value=train_nlli)]), counter)
          if val_nlli < self.best_val_nlli:
            self.best_val_nlli = val_nlli
            self.best_model_counter = counter
            self.best_model_path = curr_model_path
            # self.best_model_train_nlli = train_nlli
          # self.train_nlli_list.append(train_nlli)
          self.val_nlli_list.append(val_nlli)
          self.counter_list.append(counter)

          samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
            }
          )
          self.samples_list.append(samples)
          self.samples_list[-1].shape[1]
          manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          self.inception_list.append(curr_inception_score)
          save_images(samples, [manifold_h, manifold_w],
                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

          np.save("./"+config.sample_dir+'/graph_data', 
            [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter,\
             self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
             self.batch_train_nlli_list, self.inception_list, self.loss_list])

    
    np.save("./"+config.sample_dir+'/graph_data', 
            [self.samples_list, self.val_nlli_list, self.best_val_nlli, self.best_model_counter,\
             self.best_model_path, self.counter_list, [manifold_h, manifold_w], \
             self.batch_train_nlli_list, self.inception_list, self.loss_list])
    self.test_model(test_data, config)

  def test_model(self, test_data, config):
    print("[*] Restoring best model counter: %d, val neg lli: %.8f" 
      % (self.best_model_counter, self.best_val_nlli))
    self.saver.restore(self.sess, self.best_model_path)
    print("[*] Best model restore from: " + self.best_model_path)
    print("[*] Evaluating on the test set")
    test_nlli = self.evaluate_neg_loglikelihood(test_data, config)
    print("[*] Test negative log likelihood: %.8f" % (test_nlli))

  def calculate_inception_and_mode_score(self):
    #to get mode scores add code to load your favourite mnist classifier in inception_score.py
    if self.dataset_name == "mnist": 
      return [0.0, 0.0, 0.0, 0.0]
    sess = self.sess
    all_samples = []
    for i in range(18):
        if self.prior == "uniform":
          batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "logistic":
          batch_z = np.random.logistic(loc=0.,scale=1.0,size=[self.batch_size, self.z_dim]) \
              .astype(np.float32)
        elif self.prior == "gaussian":
          batch_z = np.random.normal(0.0, 1.0, size=(self.batch_size , self.z_dim))
        else:
          print("ERROR: Unrecognized prior...exiting")
          exit(-1)
        samples_curr = self.sess.run(
            [self.sampler],
            feed_dict={
                self.z: batch_z,}
          )
        all_samples.append(samples_curr[0])
    all_samples = np.concatenate(all_samples, axis=0)
    # return all_samples
    all_samples = (all_samples*255.).astype('int32')
    
    return inception_score.get_inception_and_mode_score(list(all_samples), sess=sess)
  
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      tf.set_random_seed(0)
      np.random.seed(0)
      if reuse:
        scope.reuse_variables()

      if self.dataset_name != "mnist":
        if self.f_div == "wgan":
          hn1 = image
         
          h0 = Layernorm('d_ln_1', [1,2,3], lrelu(conv2d(hn1, self.df_dim , name='d_h0_conv')))
          h1 = Layernorm('d_ln_2', [1,2,3], lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = Layernorm('d_ln_3', [1,2,3], lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = Layernorm('d_ln_4', [1,2,3], lrelu(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      
          return tf.nn.sigmoid(h4), h4
        else:
          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
          h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
          h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
          h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

          return tf.nn.sigmoid(h4), h4
      else:
        if self.f_div == "wgan":
          x = image

          h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))

          h1 = lrelu(conv2d(h0, self.df_dim , name='d_h1_conv'))
          h1 = tf.reshape(h1, [self.batch_size, -1])      

          h2 = lrelu(linear(h1, self.dfc_dim, 'd_h2_lin'))

          h3 = linear(h2, 1, 'd_h3_lin')

          return tf.nn.sigmoid(h3), h3
        else:
          x = image
          
          h0 = lrelu(conv2d(x, self.c_dim, name='d_h0_conv'))
          
          h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim , name='d_h1_conv')))
          h1 = tf.reshape(h1, [self.batch_size, -1])      
          
          h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
          
          h3 = linear(h2, 1, 'd_h3_lin')
            
          return tf.nn.sigmoid(h3), h3


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.input_height, self.input_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    return self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0



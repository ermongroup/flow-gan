import os
import scipy.misc
import numpy as np
np.random.seed(0)

from model import DCGAN
from utils import pp

import tensorflow as tf
tf.set_random_seed(0)

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use  [32]")
flags.DEFINE_integer("input_width", None, "The size of image to use If None, same value as input_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, multi-mnist, cifar-10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the logs [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("f_div", "wgan", "f-divergence used for specifying the objective")
flags.DEFINE_string("prior", "gaussian", "prior for generator")
flags.DEFINE_float("alpha", 1e-7, "alpha value (if applicable)")
flags.DEFINE_float("lr_decay", 1.0, "learning rate decay rate")
flags.DEFINE_float("min_lr", 0.0, "minimum lr allowed")
flags.DEFINE_float("reg", 10.0, "regularization parameter (only for wgan)")
flags.DEFINE_string("model_type", "real_nvp", "model_type")
flags.DEFINE_string("init_type", "normal", "initialization for weights")
flags.DEFINE_integer("n_critic", 1, "no of discriminator iterations")
flags.DEFINE_integer("batch_norm_adaptive", 1, "type of batch norm used (only for real-nvp)")
flags.DEFINE_integer("no_of_layers", 8,"No of units between input and output in the m function for a coupling layer")
flags.DEFINE_integer("hidden_layers", 1000, "Size of hidden layers if applicable")
flags.DEFINE_integer("gpu_nr", 0, "gpu no used")
flags.DEFINE_float("like_reg", 0, "regularizing factor for likelihood")
flags.DEFINE_integer("df_dim", 64, "Dim depth of disc")



FLAGS = flags.FLAGS

def main(_):
  np.random.seed(0)
  tf.set_random_seed(0)
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  run_config.allow_soft_placement=True
  sess = None
  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        c_dim=FLAGS.c_dim,
        z_dim=FLAGS.c_dim * FLAGS.input_height * FLAGS.input_width,
        dataset_name=FLAGS.dataset,
        checkpoint_dir=FLAGS.checkpoint_dir,
        f_div=FLAGS.f_div,
        prior=FLAGS.prior,
        lr_decay=FLAGS.lr_decay,
        min_lr=FLAGS.min_lr,
        model_type=FLAGS.model_type,
        log_dir=FLAGS.log_dir,
        alpha=FLAGS.alpha,
        batch_norm_adaptive=FLAGS.batch_norm_adaptive,
        init_type=FLAGS.init_type,
        reg=FLAGS.reg,
        n_critic=FLAGS.n_critic,
        hidden_layers=FLAGS.hidden_layers,
        no_of_layers=FLAGS.no_of_layers,
        like_reg=FLAGS.like_reg,
        df_dim=FLAGS.df_dim)

  dcgan.train(FLAGS)

if __name__ == '__main__':
  tf.app.run()

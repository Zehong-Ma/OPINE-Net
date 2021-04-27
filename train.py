'''
@File    :   train.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   training configs
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import train_model
from utils import mkdir_if_missing

slim = tf.contrib.slim

#########################
# Training Directories #
#########################

tf.app.flags.DEFINE_string('data_dir', 'data',
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string('model_dir', 'model',
                           'Directory name to save the checkpoints [checkpoint]')

tf.app.flags.DEFINE_string('log_dir', 'log',
                           'Directory name to save the logs')

#########################
#     Model Settings    #
#########################

tf.app.flags.DEFINE_float('label_smoothing', 0.0,
                          'The amount of label smoothing.')

tf.app.flags.DEFINE_integer('batch_size', 64,
                            'The number of samples in each batch.')
                        
tf.app.flags.DEFINE_integer('epochs', 220,
                            'The maximum number of training epochs.')

tf.app.flags.DEFINE_integer('num_samples', 88912,
                            'The number of samples.')

tf.app.flags.DEFINE_integer('ckpt_steps', 10000,
                            'How many steps to save checkpoints.')

tf.app.flags.DEFINE_integer('layer_num', 9,
                            'Phase number of OPINE-Net-plus')

tf.app.flags.DEFINE_integer('group_num', 1,
                            'Group number for training')

tf.app.flags.DEFINE_integer('cs_ratio', 25,
                            'From {1, 4, 10, 25, 40, 50}')

tf.app.flags.DEFINE_float('weight_decay', 0.00004,
                          'The weight decay on the model weights.')


#########################
#    Pretrain Settings  #
#########################



tf.app.flags.DEFINE_boolean("restore_last_model", False,
                            "whether to restore pretrained ckpt")

tf.app.flags.DEFINE_string("restore_path", "False",
                           "Directory name for pretrained ckpt")


###############################
#    Optimization Settings    #
###############################

tf.app.flags.DEFINE_string('optimizer', 'adam',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                           '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

tf.app.flags.DEFINE_float('adam_beta1', 0.9,
                          'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float('adam_beta2', 0.999,
                          'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-5,
                          'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('end_learning_rate', 0.000001,
                          'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float('adadelta_rho', 0.95,
                          'The decay rate for adadelta.')

tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1,
                          'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1,
                          'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float('ftrl_l1', 0.0,
                          'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float('ftrl_l2', 0.0,
                          'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float('momentum', 0.9,
                          'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9,
                          'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9,
                          'Decay term for RMSProp.')

tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential',
                           'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
                           ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          'Learning rate decay factor.')

tf.app.flags.DEFINE_float('num_epochs_per_decay', 160.0,
                          'Number of epochs after which learning rate decays.')

#########################
#   Default Settings    #
#########################

tf.app.flags.DEFINE_integer('num_gpus', 1,
                            'The number of GPUs.')

tf.app.flags.DEFINE_bool('sync_replicas', False,
                         'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1,
                            'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1,
                            'Number of worker replicas.')

tf.app.flags.DEFINE_integer('num_ps_tasks', 0,
                            'The number of parameter servers. If the value is 0, then the parameters '
                            'are handled locally by the worker.')

tf.app.flags.DEFINE_integer('task', 0,
                            'Task id of the replica running the training.')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                          'The decay to use for the moving average.'
                          'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer('input_queue_memory_factor', 16,
                            """Size of the queue of preprocessed images. """)

tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

FLAGS = tf.app.flags.FLAGS


def main(_):
    # create folders
    mkdir_if_missing(FLAGS.model_dir)
    mkdir_if_missing(FLAGS.log_dir)
    
    # training
    train_model.train()


if __name__ == '__main__':
    tf.app.run()

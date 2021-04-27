'''
@File    :   modules.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   implementation of modules(sub-networks) in the OPINE-Net 
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
ratio_dict = {1: 10, 4: 43, 10: 109, 25: 272, 30: 327, 40: 436, 50: 545}

def binary_sign(input):
    """  binary_sign(input) = 1 if input>=0 or -1 if input<0  
    """
    output = tf.where(input>=0, tf.ones_like(input, dtype=tf.float32), -1*tf.ones_like(input, dtype=tf.float32))
    return output

def sampling_subnet(image):
    with tf.variable_scope('sampling_subnet'):
        n_input = ratio_dict[FLAGS.cs_ratio]
        Phi = tf.get_variable(name='Phi',shape=[n_input,1089],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
        Phi_scale = tf.Variable(0.01,name='Phi_scale')
        Phi_ = binary_sign(Phi)
        Phi_ = Phi_scale * Phi_
        Phi_weight =tf.reshape(tf.transpose(Phi_,[1,0]), [33,33,1,n_input] )# [filter_height, filter_width, input_channel, output_channel]


        Phix = tf.nn.conv2d(input=image, filter=Phi_weight, strides=[1,33,33,1], padding='VALID', name='measurement') # check whether conv2d has bias

        return Phix, Phi_weight, Phi_


def initialization_subnet(Phix, Phi):
    with tf.variable_scope('initialization_subnet'):
        n_input = ratio_dict[FLAGS.cs_ratio]
        Phi_T_weight = tf.reshape(Phi, [1,1,n_input,1089])
        
        Phi_T_y = tf.nn.conv2d(input=Phix, filter=Phi_T_weight, strides=[1,1,1,1], padding='VALID', name='Phi_y')

        Phi_T_y = tf.depth_to_space(Phi_T_y, 33)

        x_0 = Phi_T_y
        return x_0, Phi_T_weight
    
def recovery_subnet(x_0, Phi_weight, Phi_T_weight):
    with tf.variable_scope('recovery_subnet'):
        n_input = ratio_dict[FLAGS.cs_ratio]
        x = x_0
        layers_sym = []
        for i in range(FLAGS.layer_num):
            with tf.variable_scope('recovery_phase%d'%(i+1)):
                x, layer_sym = recovery_phase(x, Phi_weight, Phi_T_weight, x_0)
                layers_sym.append(layer_sym)
        x_final = x

        return x_final, layers_sym
        




def recovery_phase(x, Phi_weight, Phi_T_weight, x_0):
    lambda_step = tf.Variable(0.5,name='lambda_step')
    soft_thr = tf.Variable(0.01,name='soft_thr') 
    x = x - lambda_step * ( PhiTPhi_fun(x, Phi_weight, Phi_T_weight) - x_0 )
    x_input = x 
    # D
    conv_D = tf.get_variable(name='conv_D_filter',shape=[3,3,1,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x_D = tf.nn.conv2d(input=x_input, filter=conv_D, strides=[1,1,1,1], padding='SAME', name='conv_D')
    # forward
    conv_1_forward = tf.get_variable(name='conv_1_forward_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x = tf.nn.conv2d(input=x_D, filter=conv_1_forward, strides=[1,1,1,1], padding='SAME', name='conv_1_forward')
    x = tf.nn.relu(x)
    conv_2_forward = tf.get_variable(name='conv_2_forward_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x_forward = tf.nn.conv2d(input=x, filter=conv_2_forward, strides=[1,1,1,1], padding='SAME', name='conv_2_forward')

    # soft
    x = tf.multiply(tf.sign(x_forward), tf.nn.relu(tf.abs(x_forward)-soft_thr),name='soft')

    # backward
    conv_1_backward = tf.get_variable(name='conv_1_backward_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x = tf.nn.conv2d(input=x, filter=conv_1_backward, strides=[1,1,1,1], padding='SAME', name='conv_1_backward')
    x = tf.nn.relu(x)
    conv_2_backward = tf.get_variable(name='conv_2_backward_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x_backward = tf.nn.conv2d(input=x, filter=conv_2_backward, strides=[1,1,1,1], padding='SAME', name='conv_2_backward')

    # G
    conv1_G = tf.get_variable(name='conv1_G_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x = tf.nn.conv2d(input=tf.nn.relu(x_backward), filter=conv1_G, strides=[1,1,1,1], padding='SAME', name='conv1_G')
    conv2_G = tf.get_variable(name='conv2_G_filter',shape=[3,3,32,32],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x = tf.nn.conv2d(input=tf.nn.relu(x), filter=conv2_G, strides=[1,1,1,1], padding='SAME', name='conv2_G')
    conv3_G = tf.get_variable(name='conv3_G_filter',shape=[3,3,32,1],initializer=tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32))
    x_G = tf.nn.conv2d(input=tf.nn.relu(x), filter=conv3_G, strides=[1,1,1,1], padding='SAME', name='conv3_G')
    
    x_pred = x_input + x_G

    # symloss
    x = tf.nn.conv2d(input=x_forward, filter=conv_1_backward, strides=[1,1,1,1], padding='SAME')
    x = tf.nn.relu(x)
    x_D_est = tf.nn.conv2d(input=x, filter=conv_2_backward, strides=[1,1,1,1], padding='SAME') 
    symloss = x_D_est - x_D
    
    return x_pred, symloss


def PhiTPhi_fun(x, Phi_weight, Phi_T_weight):
    temp = tf.nn.conv2d(x, filter=Phi_weight, strides=[1,33,33,1], padding='VALID')
    temp = tf.nn.conv2d(temp, filter=Phi_T_weight, strides=[1,1,1,1], padding='VALID')
    return tf.depth_to_space(temp, 33)




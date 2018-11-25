# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 16:30:57 2018

@author: hty
"""
import tensorflow as tf
import h5py
import numpy as np

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (tf.reduce_mean(tf.square(y_pred - y_true)))) 

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32 )
    return tf.Variable(initial)
    
def conv2d(x, W,stride=[1,1,1,1],pad='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)
    
def relu(x):
    return tf.nn.relu(x)

def read_data(path):
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    train_data = np.transpose(data, (0, 2, 3, 1))
    train_label = np.transpose(label, (0, 2, 3, 1))
    print(train_data.shape)
    print(train_label.shape)
    return train_data, train_label

    
def vdsr(input_img,pad='SAME'):
    ksize = 64
    
    w1 = tf.get_variable('conv1', shape=[3, 3, 1, ksize], initializer=tf.contrib.keras.initializers.he_normal())
    x = relu(conv2d(input_img,w1,[1,1,1,1],pad))
    
    for i in range(18):
        name = 'conv' + str(i+2)
        w = tf.get_variable(name, shape=[3, 3, ksize, ksize], initializer=tf.contrib.keras.initializers.he_normal())
        x = relu(conv2d(x,w,[1,1,1,1],pad))
    
    w_final = tf.get_variable('conv20',shape=[3, 3, ksize, 1], initializer=tf.contrib.keras.initializers.he_normal())
    x = conv2d(x,w_final,[1,1,1,1],pad)
    output = x + input_img
   
    return output
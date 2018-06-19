import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=False, validation_size=0)
train_x = mnist.train.images


def discriminator(x, isTrain=True, reuse=False, with_Q=False):

    with tf.variable_scope(name_or_scope='Dis', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 64, 4, (2, 2), 'same', activation=tf.nn.leaky_relu)
        conv2 = tf.layers.conv2d(conv1, 128, 4, (2, 2), 'same', activation=tf.nn.leaky_relu, use_bias=False)
        conv2 = tf.layers.batch_normalization(conv2, training=isTrain)
        dense1 = tf.layers.dense(conv2, 1024, tf.nn.leaky_relu, False)
        dense1 = tf.layers.batch_normalization(dense1, training=isTrain)
        D = tf.layers.dense(dense1, 1)  # logit

        if not with_Q:
            return D

        else:
            dense2 = tf.layers.dense(dense1, 128, None, False)
            dense2 = tf.layers.batch_normalization(dense2, training=isTrain)
            dense2 = tf.nn.leaky_relu(dense2)
            Q = tf.layers.dense(dense2, 1)  # logit
            return D, Q


def generator(z, c, isTrain=True, reuse=False):

    with tf.variable_scope(name_or_scope='Gen', reuse=reuse):
        noise = tf.concat([z, c], 0)
        dense1 = tf.layers.dense(noise, 1024, tf.nn.relu, False)
        dense1 = tf.layers.batch_normalization(dense1, training=isTrain)
        dense2 = tf.layers.dense(dense1, 7 * 7 * 128, tf.nn.relu, False)
        dense2 = tf.layers.batch_normalization(dense2, training=isTrain)
        dense2 = tf.reshape(dense2, [None, 7, 7, 128])
        conv1 = tf.layers.conv2d_transpose(dense2, 64, 4, (2, 2), 'same', activation=tf.nn.relu, use_bias=False)
        conv1 = tf.layers.batch_normalization(conv1, training=isTrain)
        conv2 = tf.layers.conv2d_transpose(conv1, 1, 4, (2, 2), 'same')  # 논문엔 따로 얘기 없지 stride 2여야 됨

        return conv2

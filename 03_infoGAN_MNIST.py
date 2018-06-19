import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=False, validation_size=0)
train_x = mnist.train.images

batch_size = 100
learning_rate = 2e-4
training_epoch = 20
beta1 = .5

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
            Q = tf.layers.dense(dense2, 12)  # logit
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


def random_noise(batch_size):
    return np.random.normal(0, 1, size=[batch_size, 62])


def random_condition(batch_size):
    categorical = np.random.multinomial(1, [1/10] * 10, batch_size)
    continuous = np.random.uniform(-1, 1, size=[batch_size, 2])
    return np.concatenate([categorical, continuous], axis=1)


global_time0 = time.time()
g = tf.Graph()
with g.as_default():

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Z = tf.placeholder(tf.float32, [None, 62])
    C = tf.placeholder(tf.float32, [None, 12])
    isTrain = tf.placeholder(tf.bool)

    fake_x = generator(Z, C)
    fake_logits, Q_logits = discriminator(fake_x, isTrain=isTrain, with_Q=True)
    real_logits = discriminator(X, isTrain=isTrain, reuse=True)


    g_loss_base = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits,
            labels=tf.ones([batch_size, 1])
        )
    )
    information_regularizer = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Q_logits,
            labels=tf.zeros([batch_size, 1])
        )
    ) +
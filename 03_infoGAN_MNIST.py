import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from functools import partial


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=False, validation_size=0)
train_x = mnist.train.images

batch_size = 256
learning_rate = 2e-4
training_epoch = 30
beta1 = .5

dir_name = 'infoGAN'
save_path = dir_name + '/Checkpoints/'
summary_path = dir_name + '/Summaries'
image_path = dir_name + '/Images/'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    for f in os.listdir(save_path):
        os.remove(save_path + f)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)
else:
    for f in os.listdir(summary_path):
        os.remove(summary_path + '/' + f)

if not os.path.exists(image_path):
    os.makedirs(image_path)
else:
    for f in os.listdir(image_path):
        os.remove(image_path + f)


def discriminator(x, isTrain=True, reuse=False, with_Q=False):

    lrelu = partial(tf.nn.leaky_relu, alpha=.1)
    BN = partial(tf.layers.batch_normalization, training=isTrain, momentum=.9)
    BN_lrelu = lambda x: lrelu(BN(x))

    with tf.variable_scope(name_or_scope='Dis', reuse=reuse):

        conv1 = tf.layers.conv2d(x, 64, 4, (2, 2), 'same', activation=lrelu, use_bias=True)
        conv2 = tf.layers.conv2d(conv1, 128, 4, (2, 2), 'same', activation=BN_lrelu, use_bias=False)
        flat1 = tf.layers.flatten(conv2)
        dense1 = tf.layers.dense(flat1, 1024, BN_lrelu, False)
        D = tf.layers.dense(dense1, 1)  # logit

        if not with_Q:
            return D

        else:
            with tf.variable_scope(name_or_scope='Con', reuse=reuse):

                dense2 = tf.layers.dense(dense1, 128, BN_lrelu, False)
                Q = tf.layers.dense(dense2, 12)  # logit

            return D, Q


def generator(z, c, isTrain=True, reuse=False):

    BN = partial(tf.layers.batch_normalization, training=isTrain, momentum=.9)
    BN_relu = lambda x: tf.nn.relu(BN(x))

    with tf.variable_scope(name_or_scope='Gen', reuse=reuse):

        noise = tf.concat([z, c], 1)

        dense1 = tf.layers.dense(noise, 1024, BN_relu, False)
        dense2 = tf.layers.dense(dense1, 7 * 7 * 128, BN_relu, False)
        patch = tf.reshape(dense2, [-1, 7, 7, 128])  # None 하면 에러남
        conv1 = tf.layers.conv2d_transpose(patch, 64, 4, (2, 2), 'same', activation=BN_relu, use_bias=False)
        conv2 = tf.layers.conv2d_transpose(conv1, 1, 4, (2, 2), 'same', activation=tf.nn.sigmoid, use_bias=True)  # 논문엔 따로 얘기 없지만 stride 2여야 됨

        return conv2


def random_noise(batch_size):
    return np.random.uniform(-1, 1, size=[batch_size, 62])


def random_code(batch_size):
    categorical = random_categorical(batch_size)
    continuous = random_continuous(batch_size)
    return np.concatenate([categorical, continuous], axis=1)


def random_categorical(batch_size):
    return np.random.multinomial(1, [1/10] * 10, batch_size)


def random_continuous(batch_size):
    return np.random.uniform(-1, 1, size=[batch_size, 2])


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

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_logits,
            labels=tf.ones_like(real_logits)
        )
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits,
            labels=tf.zeros_like(fake_logits)
        )
    )
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_logits,
            labels=tf.ones_like(fake_logits)
        )
    )

    q_loss_disc = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=Q_logits[:, :10],
            labels=C[:, :10]
        )
    )  # categorical은 확률 space이고 continuous는 아니기 때문에 continuous는 logit을 그대로 씀
    q_loss_cont = tf.reduce_mean(tf.reduce_sum(
        tf.square(Q_logits[:, 10:] - C[:, 10:]),
        axis=1)
    )
    q_loss = q_loss_disc + q_loss_cont

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'Gen' in var.name]
    d_vars = [var for var in t_vars if 'Dis' in var.name]
    q_vars = [var for var in t_vars if any(x in var.name for x in ['Gen', 'Dis', 'Con'])]

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

    d_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1).minimize(g_loss, var_list=g_vars)
    q_opt = tf.train.AdamOptimizer(learning_rate * 5, beta1=beta1).minimize(q_loss, var_list=q_vars)

    saver = tf.train.Saver(max_to_keep=1)

    with tf.name_scope('Generator'):
        summary_g = tf.summary.merge(
            [tf.summary.scalar('g_loss', g_loss), tf.summary.scalar('d_loss_fake', d_loss_fake)])
    with tf.name_scope('Discriminator'):
        summary_d = tf.summary.merge(
            [tf.summary.scalar('d_loss', d_loss), tf.summary.scalar('d_loss_real', d_loss_real)])
    with tf.name_scope('Auxiliary Distribution'):
        summary_q = tf.summary.merge(
            [tf.summary.scalar('q_loss', q_loss),
             tf.summary.scalar('q_loss_disc', q_loss_disc),
             tf.summary.scalar('q_loss_cont', q_loss_cont)])
    writer = tf.summary.FileWriter(summary_path)

with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())
    total_batches = len(train_x) // batch_size
    plt.set_cmap('inferno')  # 추천 : gray, inferno, RdGy_r

    sample_noise = random_noise(50)
    # 0, 1, 2, 3, 4 one-hot vector 각각 열 개씩
    sample_categorical = np.eye(10)[(np.ones(50, np.int32).reshape(-1, 5) * np.arange(5)).T.flatten()]
    sample_continuous = random_continuous(50)
    sample_code = np.concatenate([sample_categorical, sample_continuous], axis=1)

    global_step = 0

    for epoch in range(training_epoch):

        epoch_time0 = time.time()

        for batch in range(total_batches):

            t0 = time.time()

            noise = random_noise(batch_size)
            code = random_code(batch_size)
            batch_x = train_x[batch * batch_size:(batch + 1) * batch_size]
            sess.run(d_opt, feed_dict={X: batch_x, Z: noise, C: code, isTrain: True})
            sess.run(g_opt, feed_dict={Z: noise, C: code, isTrain: True})
            sess.run(q_opt, feed_dict={X: batch_x, Z: noise, C: code, isTrain: True})

            t1 = time.time()

            gl, dl, ql = sess.run(
                [g_loss, d_loss, q_loss],
                feed_dict={X: batch_x, Z: noise, C: code, isTrain: False})

            if global_step % 10 == 0:
                gs, ds, qs = sess.run([summary_g, summary_d, summary_q],
                                   feed_dict={X: batch_x, Z: noise, C: code, isTrain: False})
                writer.add_summary(gs, global_step)
                writer.add_summary(ds, global_step)
                writer.add_summary(qs, global_step)

            print('epoch : {:02d}/{:02d}'.format(epoch + 1, training_epoch), end=' | ')
            print('batch : {:03d}/{:03d}'.format(batch + 1, total_batches), end=' | ')
            print('time spent : {:.3f} sec'.format(t1 - t0), end=' | ')
            print('gl : {:.6f} | dl : {:.6f} | ql : {:.6f}'.format(gl, dl, ql))

            global_step += 1

        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        print('==================== EPOCH :', epoch + 1, ' has ended ====================')
        print('Generator Loss : {:.6f}\nDiscriminator Loss : {:.6f}'.format(gl, dl))
        print('Auxiliary Distribution Loss : {:.6f}'.format(ql))
        print('Time Spent in This Epoch : {:.3f}'.format(epoch_time))

        generated = sess.run(fake_x, feed_dict={Z: sample_noise, C: sample_code, isTrain: False})
        generated = np.clip(generated, 0., .8)  # inferno 색이 너무 강렬해서 ㅎ
        fig, ax = plt.subplots(5, 10, figsize=(10, 5))  # figsize : 가로, 세로 크기 (인치)
        for i in range(5):
            for j in range(10):
                ax[i, j].set_axis_off()
                ax[i, j].imshow(generated[i*10+j, :, :, 0])
        fig.suptitle('EPOCH : {:02d}'.format(epoch+1), verticalalignment='baseline', fontweight='bold')
        plt.savefig(image_path + '{:02d}.png'.format(epoch+1), bbox_inches='tight')
        plt.close(fig)

        saver.save(sess, save_path + 'model', global_step=epoch)

    global_time1 = time.time()
    global_time_spent = global_time1 - global_time0
    print(('=' * 40 + '\n') * 3, 'Optimization Has Been Completed!!')
    print('Total Time Spent : {:.3f}'.format(global_time_spent))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', reshape=False, validation_size=0)

train_x = mnist.train.images

kernel_initializer = None
gen_activation = tf.nn.leaky_relu
dis_activation = tf.nn.leaky_relu

batch_size = 100
learning_rate = 2e-4
training_epoch = 20
beta1 = .5


def generator(z, isTrain=True, reuse=False):
    # z : [None, 1, 1, 100]
    with tf.variable_scope(name_or_scope='Gen', reuse=reuse):
        conv1 = tf.layers.conv2d_transpose(z, filters=1024, kernel_size=4, strides=(1, 1),
                                           padding='valid',
                                           kernel_initializer=kernel_initializer)  # 4, 4, 1024
        relu1 = gen_activation(tf.layers.batch_normalization(conv1, training=isTrain))
        conv2 = tf.layers.conv2d_transpose(relu1, filters=512, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_initializer=kernel_initializer)  # 8, 8, 512
        relu2 = gen_activation(tf.layers.batch_normalization(conv2, training=isTrain))
        conv3 = tf.layers.conv2d_transpose(relu2, filters=256, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_initializer=kernel_initializer)  # 16, 16, 256
        relu3 = gen_activation(tf.layers.batch_normalization(conv3, training=isTrain))
        conv4 = tf.layers.conv2d_transpose(relu3, filters=128, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_initializer=kernel_initializer)  # 32, 32, 128
        relu4 = gen_activation(tf.layers.batch_normalization(conv4, training=isTrain))
        conv5 = tf.layers.conv2d_transpose(relu4, filters=1, kernel_size=4, strides=(2, 2),
                                           padding='same',
                                           kernel_initializer=kernel_initializer)  # 64, 64, 1
        return tf.nn.tanh(conv5)


def discriminator(x, isTrain=True, reuse=False):
    # x : [None, 64, 64, 1]
    with tf.variable_scope(name_or_scope='Dis', reuse=reuse):
        conv1 = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=kernel_initializer)  # 32, 32, 128
        relu1 = dis_activation(conv1)
        conv2 = tf.layers.conv2d(relu1, filters=256, kernel_size=4, strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=kernel_initializer)  # 16, 16, 256
        relu2 = dis_activation(tf.layers.batch_normalization(conv2, training=isTrain))
        conv3 = tf.layers.conv2d(relu2, filters=512, kernel_size=4, strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=kernel_initializer)  # 8, 8, 512
        relu3 = dis_activation(tf.layers.batch_normalization(conv3, training=isTrain))
        conv4 = tf.layers.conv2d(relu3, filters=1024, kernel_size=4, strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=kernel_initializer)  # 4, 4, 1024
        relu4 = dis_activation(tf.layers.batch_normalization(conv4, training=isTrain))
        conv5 = tf.layers.conv2d(relu4, filters=1, kernel_size=4, strides=(1, 1),
                                 padding='valid',
                                 kernel_initializer=kernel_initializer)  # 1, 1, 1
        return conv5


def random_noise(batch_size):
    return np.random.normal(0, 1, size=[batch_size, 1, 1, 100])


global_time0 = time.time()

g = tf.Graph()

with g.as_default():

    X = tf.placeholder(tf.float32, [None, 64, 64, 1])
    Z = tf.placeholder(tf.float32, [None, 1, 1, 100])
    isTrain = tf.placeholder(tf.bool)

    fake_x = generator(Z)
    result_of_fake = discriminator(fake_x, isTrain=isTrain)
    result_of_real = discriminator(X, isTrain=isTrain, reuse=True)

    epsilon = 1e-12
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_real, labels=tf.ones([batch_size, 1, 1, 1])))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake, labels=tf.zeros([batch_size, 1, 1, 1])))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake,
                                                                    labels=tf.ones([batch_size, 1, 1, 1])))

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'Gen' in var.name]
    d_vars = [var for var in t_vars if 'Dis' in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1)

    g_opt = optimizer.minimize(g_loss, var_list=g_vars)
    d_opt = optimizer.minimize(d_loss, var_list=d_vars)

with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())
    total_batches = len(train_x) // batch_size
    sample_noise = random_noise(10)

    train_x = tf.image.resize_images(train_x, [64, 64]).eval()
    train_x = 2 * train_x - 1

    for epoch in range(training_epoch):

        epoch_time0 = time.time()

        for batch in range(total_batches):

            t0 = time.time()

            batch_x = train_x[batch*batch_size:(batch+1)*batch_size]
            noise = random_noise(batch_size)
            sess.run(d_opt, feed_dict={X : batch_x, Z : noise, isTrain : True})
            noise = random_noise(batch_size)
            sess.run(g_opt, feed_dict={Z : noise, isTrain : True})
            gl, dl = sess.run([g_loss, d_loss], feed_dict={X : batch_x, Z : noise, isTrain : False})

            t1 = time.time()
            print('epoch : {:02d}/{:02d}'.format(epoch+1, training_epoch), end=' | ')
            print('batch : {:03d}/{:03d}'.format(batch+1, total_batches), end=' | ')
            print('time spent : {:.3f} sec'.format(t1-t0), end=' | ')
            print('gl : {:.6f} | dl : {:.6f}'.format(gl, dl))

        epoch_time1 = time.time()
        epoch_time = epoch_time1 - epoch_time0
        print('==================== EPOCH :', epoch+1, ' ended ====================')
        print('Generator Loss : {:.6f}\nDiscriminator Loss : {:.6f}'.format(gl, dl))
        print('Time Spent in This Epoch : {:.3f}'.format(epoch_time))

        generated = sess.run(fake_x, feed_dict={Z : sample_noise, isTrain : False}) / 2 + .5
        # generated = (generated + 1) * 255 / 2
        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(generated[i,:,:,0])
        plt.savefig('dcgan-generated/{}.png'.format(str(epoch+1).zfill(3)), bbox_inches='tight')
        plt.close(fig)

    global_time1 = time.time()
    global_time_spent = global_time1 - global_time0
    print(('=' * 40 + '\n') * 3, 'Optimization Has Been Completed!!')
    print('Total Time Spent : {:.3f}'.format(global_time_spent))
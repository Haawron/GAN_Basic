import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

train_x = mnist.train.images

total_epochs = 100
batch_size = 100
learning_rate = 2e-3

initializer = tf.random_normal_initializer(mean=.0, stddev=.01)

def generator(z, reuse=False):

    with tf.variable_scope(name_or_scope='Gen', reuse=reuse):
        gw1 = tf.get_variable('w1', [128, 256], initializer=initializer)
        gb1 = tf.get_variable('b1', [256],      initializer=initializer)
        gw2 = tf.get_variable('w2', [256, 784], initializer=initializer)
        gb2 = tf.get_variable('b2', [784],      initializer=initializer)

    hidden = tf.nn.leaky_relu(tf.matmul(z, gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden, gw2) + gb2)

    return output

def discriminator(x, reuse=False):

    with tf.variable_scope(name_or_scope='Dis', reuse=reuse):
        dw1 = tf.get_variable('w1', [784, 256], initializer=initializer)
        db1 = tf.get_variable('b1', [256],      initializer=initializer)
        dw2 = tf.get_variable('w2', [256, 1],   initializer=initializer)
        db2 = tf.get_variable('b2', [1],        initializer=initializer)

    hidden = tf.nn.leaky_relu(tf.matmul(x, dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden, dw2) + db2)

    return output

def random_noise(batch_size):
    return np.random.normal(size=[batch_size, 128])


g = tf.Graph()

with g.as_default():

    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, 128])

    fake_x = generator(Z)

    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X, True)

    g_loss = tf.reduce_mean(tf.log(result_of_fake + epsilon))
    d_loss = tf.reduce_mean(tf.log(result_of_real + epsilon) + tf.log(1 - result_of_fake + epsilon))

    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'Gen' in var.name]
    d_vars = [var for var in t_vars if 'Dis' in var.name]

    optimizer = tf.train.RMSPropOptimizer(learning_rate)

    g_opt = optimizer.minimize(-g_loss, var_list=g_vars)
    d_opt = optimizer.minimize(-d_loss, var_list=d_vars)


with tf.Session(graph=g) as sess:

    sess.run(tf.global_variables_initializer())

    total_batches = int(len(train_x) / batch_size)

    sample_noise = random_noise(10)

    for epoch in range(total_epochs):

        for batch in range(total_batches):

            batch_x = train_x[np.random.random_integers(0, len(train_x), batch_size)]
            noise = random_noise(batch_size)

            sess.run(g_opt, feed_dict={Z : noise})
            sess.run(d_opt, feed_dict={X : batch_x, Z : noise})
            g_check, d_check = sess.run([g_loss, d_loss], feed_dict={X : batch_x, Z : noise})

        print('==================== EPOCH :', epoch, '====================')
        print('Generator Loss : {}\nDiscriminator Loss : {}'.format(g_check, d_check))

        if epoch % 5 == 4 or epoch == 0:
            generated = sess.run(fake_x, feed_dict={Z : sample_noise})

            fig, ax = plt.subplots(1, 10, figsize=(10, 1))
            for i in range(10):
                ax[i].set_axis_off()
                ax[i].imshow(np.reshape(generated[i], (28, 28)))

            plt.savefig('haawron-mnist-gan/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

    print('========================================\n' * 3, 'Optimization Completed!!')
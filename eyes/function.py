import tensorflow as tf
import numpy as np


def conv(input_image, kernel_size, stride, out_channels, name, padding):
    # 得到输入图片的通道数
    input_channels = input_image.get_shape()[-1]
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1], input_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.1))
        x = tf.nn.conv2d(input_image, w, stride, padding=padding, name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x


def pool(input_image, kernel, stride, name, padding):
    x = tf.nn.max_pool(input_image, ksize=kernel, strides=stride, padding=padding, name=name)
    return x


def FC_layer(input_image, out_nodes, name):
    shape = input_image.get_shape()
    if (len(shape) == 4):
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(input_image, [-1, size])
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


def loss(logits, labels):
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope + '/loss', loss)
        return loss


def optimize(loss, learning_rate, global_step):
    with tf.name_scope('optimize'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return train_op


def accuracy(logits, labels):
    with tf.name_scope('accuracy')as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0
        tf.summary.scalar(scope + '/accuracy', accuracy)
        return accuracy

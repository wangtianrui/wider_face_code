import tensorflow as tf
import function


def fcn_net(image, num_class):  # image:[227,227,3]
    with tf.name_scope('fcn_net'):
        with tf.name_scope('floor_1'):
            conv1 = function.conv(image, kernel_size=[3, 3], stride=[1, 1, 1, 1],
                                  out_channels=16, name='conv1', padding='SAME')
            pool1 = function.pool(conv1, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], name='pool1', padding='SAME')

            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm1')
        with tf.name_scope('floor_2'):
            conv2 = function.conv(norm1, kernel_size=[3, 3], stride=[1, 1, 1, 1],
                                  out_channels=16, name='conv2', padding='SAME')
            norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                              beta=0.75, name='norm2')
            pool2 = function.pool(norm2, kernel=[1, 3, 3, 1], stride=[1, 2, 2, 1], name='pool2', padding='VALID')
            print(pool2)
        with tf.name_scope('cnov_final'):
            softmax_linear = function.conv(pool2,kernel_size=[1,51,51,1],stride=[1,1,1,1],out_channels=16,name='out',padding='VALID')
        return softmax_linear

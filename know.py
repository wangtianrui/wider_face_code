import tensorflow as tf

# help(tf.contrib.layers.xavier_initializer)
# with tf.Session():
#     x = tf.get_variable('x',shape=[2,4],initializer=tf.constant_initializer())
#     x.initializer.run()
#     print(x.eval())
from numpy.ma.tests.test_core import A

input = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]

with tf.Session:
    a = tf.Variable([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    b = tf.Variable([[1, 2], [3, 4]])
    conv = tf.nn.conv2d(input=a, filter=b, strides=[1, 1, 1, 1], padding="VALID")
    print(conv.eval())

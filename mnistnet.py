import tensorflow as tf




def conv2d(x, kernel, out_channels,name):
    input_channels = x.get_shape()[-1]
    # strides=[1,x_movement,y_movement,1]
    with tf.variable_scope(name):
        w = tf.get_variable(name='weights',
                            shape=[kernel[0], kernel[1], input_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.1))
    return tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

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

def net(image, num_class):
    with tf.name_scope('alex_net'):
        with tf.name_scope('floor_1'):
            conv1 = conv2d(image,[5,5],32,name="conv1")
            pool1 = max_pool_2x2(conv1)
        with tf.name_scope('floor_2'):
            conv2 = conv2d(pool1,[5,5],64,name="conv2")
            pool2 = max_pool_2x2(conv2)
        with tf.name_scope('fc'):
            fc_1 = FC_layer(pool2, 1024, "fc1")
            fc_1_drop_out = tf.nn.dropout(fc_1,0.5)
            softmax_linear = FC_layer(fc_1_drop_out, num_class, 'soft_max')
        return softmax_linear

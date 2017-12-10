import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


def read_and_decode(tfrecords_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecords_file])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)

    image = tf.reshape(image, [208, 208, 3])
    label = tf.cast(img_features['label'], tf.int32)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=64,
                                                      capacity=20000,
                                                      min_after_dequeue=3000)
    return image_batch, tf.reshape(label_batch, [batch_size])


def plot_images(images, labels):
    '''plot one batch size
    '''
    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(str(labels[i]), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()


tfrecords_file = 'F:/Traindata/faceTF/208x208(2).tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=25)

with tf.Session()  as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:
            image, label = sess.run([image_batch, label_batch])
            plot_images(image, label)
            i += 1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)

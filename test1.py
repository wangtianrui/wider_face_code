import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

tf.device('/gpu:0')
BATCH_SIZE = 25
img_w = 208
img_h = 208


# %%

def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            # print("name:",root)
            images.append(root + '/' + name)
        # get 10 sub-folder names
        for name in sub_folders:
            temp.append(os.path.join(root, name))

    # assign 10 labels based on the folder names
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('/')[-1]
        # print("letter test",letter)
        if letter == 'activedata':
            labels = np.append(labels, n_img * [1])
        elif letter == 'negativedata':
            labels = np.append(labels, n_img * [0])

        else:
            labels = np.append(labels, n_img * [10])

    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])  # 图片绝对路径
    label_list = list(temp[:, 1])  # label
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


# %%

def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# %%

def convert_to_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)

    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' % (images.shape[0], n_samples))

    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            # image[i] = image[i]
            print("path test:", images[i])
            if os.path.exists(images[i]) == False:
                continue

            image = cv2.imread(images[i])  # type(image) must be array!
            res = cv2.resize(image, (208, 208), interpolation=cv2.INTER_CUBIC)
            # cv2.imshow(res)
            # cv2.waitKeyEx(0)
            image_raw = res.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': int64_feature(label),
                'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' % e)
            print('Skip it!\n')
        except AttributeError as e:
            continue
            print(e)
        except cv2.error:
            continue
    writer.close()
    print('Transform done!')


# %%
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
    label = tf.cast(img_features['label'], tf.int32)
    '''
    image = tf.reshape(image, [208, 208, 3])
    image = tf.cast(image, tf.float32)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=2000)
    '''
    image.set_shape([208*208*3])
    image = tf.reshape(image, [208, 208, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    cv2.imshow('test',image)
    plt.subplot(5, 5, i + 1)
    plt.axis('off')
    plt.title(str(label), fontsize=14)
    plt.subplots_adjust(top=1.5)
    print(image)
    plt.imshow(image)
    plt.show()
    #return image_batch, tf.reshape(label_batch, [batch_size])
tfrecords_file = 'F:/Traindata/faceTF/208x208(2).tfrecords'
op = read_and_decode(tfrecords_file=tfrecords_file,batch_size=1)

sess = tf.Session()
sess.run(op)
'''
# %% Convert data to TFRecord

test_dir = 'C://Users//Windows7//Documents//Python Scripts//notMNIST//notMNIST_small//'
save_dir = 'D://python_code//03 TFRecord//save//'
BATCH_SIZE = 25

# Convert test data: you just need to run it ONCE !
name_test = 'test'
images, labels = get_file(test_dir)
convert_to_tfrecord(images, labels, save_dir, name_test)


# %% TO test train.tfrecord file

def plot_images(images, labels):

    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(chr(ord('A') + labels[i] - 1), fontsize=14)
        plt.subplots_adjust(top=1.5)
        plt.imshow(images[i])
    plt.show()

'''


def plot_images(images, labels):
    for i in np.arange(0, BATCH_SIZE):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title(str(labels[i]), fontsize=14)
        plt.subplots_adjust(top=1.5)
        print(images[i])
        plt.imshow(images[i])
    plt.show()


'''
tfrecords_file = 'F:/Traindata/faceTF/208x208(2).tfrecords'
image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
plot_images(image_batch, label_batch)


with tf.Session()  as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i < 1:
            # just plot
            # one batch size
            image, label = sess.run([image_batch, label_batch])
            plot_images(image, label)
            i += 1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
'''
# test_dir = 'F:/Traindata/facedata/'
# save_dir = 'F:/Traindata/faceTF/'
# images, labels = get_file(test_dir)
# convert_to_tfrecord(images, labels, save_dir, '208x208')

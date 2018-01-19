import tensorflow as tf
import os
import numpy as np
import function
import alex_net
import input
import vgg_net
import fcn_net
import cv2
import cifar_net
import mnistnet

NUM_CLASS = 2
# IMG_W = 208
# IMG_H = 208
IMG_W = 24
IMG_H = 24
BATCH_SIZE = 33
LEARNING_RATE = 0.0001
MAX_STEP = 15000
CAPACITY = 2000
#testCifar10 = 'E:/python_programes/datas/cifar10/cifar-10-batches-bin/'
# TRAIN_PATH = 'F:/Traindata/faceTF/208x208(2).tfrecords'
TRAIN_PATH = 'F:/Traindata/eyes/eyes.tfrecords'
TEST_PATH = 'F:/Traindata/eyes/eyestest.tfrecords'
# train_log_dir = 'E:/python_programes/trainRES/face_wide_res/'
train_log_dir = 'F:/Traindata/eyes/result/'
tf.device("/gpu:0")


def train():
    with tf.name_scope('input'):
        # train_image_batch, train_labels_batch = input.read_cifar10(TRAIN_PATH, batch_size=BATCH_SIZE)
        train_image_batch, train_labels_batch = input.read_and_decode_by_tfrecorder(TRAIN_PATH, BATCH_SIZE)
        test_image_batch,test_labels_batch = input.read_and_decode_by_tfrecorder(TEST_PATH,BATCH_SIZE)
        print(train_image_batch)
        print(train_labels_batch)
        # show = cv2.imshow('test',train_image_batch[0])
        # wait = cv2.waitKeyEx()

        #logits = alex_net.alex_net(train_image_batch, NUM_CLASS)
        # logits = fcn_net.fcn_net(train_image_batch,NUM_CLASS)
        logits = cifar_net.inference(train_image_batch, batch_size=BATCH_SIZE, n_classes=NUM_CLASS,name="train")
        # logits = VGG.VGG16N(train_image_batch,n_classes=NUM_CLASS,is_pretrain=False)
        #logits = mnistnet.net(train_image_batch,num_class=NUM_CLASS)
        print(logits)
        loss = function.loss(logits=logits, labels=train_labels_batch)
        accuracy_logits = cifar_net.inference(test_image_batch,batch_size=BATCH_SIZE,n_classes=NUM_CLASS,name="test")
        accuracy = function.accuracy(logits=accuracy_logits, labels=test_labels_batch)

        my_global_step = tf.Variable(0, name='global_step')
        train_op = function.optimize(loss=loss, learning_rate=LEARNING_RATE, global_step=my_global_step)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):

            if coord.should_stop():
                break

            _, train_loss, train_accuracy = sess.run([train_op, loss, accuracy])
            #print('***** Step: %d, loss: %.4f *****' % (step, train_loss))
            if (step % 50 == 0) or (step == MAX_STEP):
                print('***** Step: %d, loss: %.4f' % (step, train_loss))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
            if (step % 200 == 0) or (step == MAX_STEP):
                print('***** Step: %d, loss: %.4f, accuracy: %.4f%% *****' % (step, train_loss, train_accuracy))
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or step == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        print('error')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()


train()

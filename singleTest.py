import tensorflow as tf
import cifar_net
import function
import numpy as np
import cv2


def image_read(filename1):
    if not tf.gfile.Exists(filename1):
        print("filname1 does not exists")

    """
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    image = tf.image.decode_jpeg(image_data, channels=3)
    # image = tf.ones(shape=[24,24,3],name='input')
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [1, 24, 24, 3])
    """
    image1 = cv2.imread(filename=filename1)
    image1 = cv2.resize(image1, (24, 24), interpolation=cv2.INTER_CUBIC)
    image1 = tf.reshape(image1, [1,24, 24, 3])
    image1 = tf.image.convert_image_dtype(image1, tf.int32)
    # image2 = tf.Variable
    holder = tf.ones(shape=[1,24, 24, 3], dtype=tf.int32, name="ones")
    # image2 = cv2.resize(image2, (24, 24), interpolation=cv2.INTER_CUBIC)
    # image2 = tf.image.convert_image_dtype(image2, tf.float32)
    # image2 = tf.reshape(image2, [1, 24, 24, 3])
    # print(image_data)
    # 将字符串转换成float
    image = tf.concat( [holder, image1],0)
    image = tf.image.convert_image_dtype(image, tf.float32)
    print(image)
    return image


def eval(image):
    logits = cifar_net.inference(image, 2, 2)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("F:/Traindata/eyes/result/")
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            sess.run(init_op)
            # saver = tf.train.import_meta_graph("data/data.ckpt.meta")
            # saver.restore(sess, 'cifar10_trainresults/data.chkp.data-00000-of-00001')
            saver.restore(sess, ckpt.model_checkpoint_path)
            # saver.restore(sess, "logs/train/model.ckpt")

        predict = tf.argmax(logits, dimension=1, name='output')
        print(predict.eval())
        print(logits.eval())


def main(argv=None):
    filename = './testEyes/1 (2).jpg'
    img = image_read(filename)
    eval(img)


if __name__ == '__main__':
    tf.app.run()

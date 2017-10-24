import tensorflow as tf
import numpy as np
import input
import fundus

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_imgs = mnist.train.images.reshape([-1,28,28,1])
train_labs = mnist.train.labels
test_imgs = mnist.test.images.reshape([-1,28,28,1])
test_labs = mnist.test.labels

debug_test=True

#train_images, train_labels, train_filenames, test_images, test_labels, test_filenames=fundus.type1('./fundus_300_debug' , resize=(288,288))
test_images=test_imgs/255.
sess = tf.Session()
saver = tf.train.import_meta_graph(meta_graph_or_file='./models/mnist/model-38100.meta')
saver.restore(sess, save_path='./models/mnist/model-38100')

x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
is_training_=tf.get_default_graph().get_tensor_by_name('is_training_:0')

pred=sess.run(fetches=[pred_] ,feed_dict={x_ : test_imgs[:30] , is_training_ : True})
print pred


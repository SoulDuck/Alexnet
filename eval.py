import tensorflow as tf
import numpy as np
import input
import fundus

train_images, train_labels, train_filenames, test_images, test_labels, test_filenames=fundus.type1('./fundus_300_debug' , resize=(288,288))
test_images=test_images/255.
sess = tf.Session()
saver = tf.train.import_meta_graph(meta_graph_or_file='./models/model-199900.meta')
saver.restore(sess, save_path='./models/model-199900')

x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
is_training_=tf.get_default_graph().get_tensor_by_name('Placeholder:0')

pred=sess.run(fetches=[pred_] ,feed_dict={x_ : test_images[:30] , is_training_ : True})
print pred


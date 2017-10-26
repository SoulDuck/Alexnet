import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
"""
#tf cond usage 

is_training=tf.placeholder(tf.bool , [])
def _fn1():
    print 'a'
    return 'a'
def _fn2():
    print 'b'
    return tf.Variable('4')

conv_keep_prob = tf.cond(is_training ,lambda: _fn1(), lambda: _fn2())
sess=tf.Session()
init=tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
sess.run(init)
b=sess.run(fetches=conv_keep_prob , feed_dict={is_training : False})
"""

test_imgs=np.load('test_images.npy')
test_imgs=np.asarray(test_imgs)
print np.max(test_imgs)
print np.min(test_imgs)
print test_imgs[0]

print np.shape(test_imgs[0])
img=Image.fromarray(test_imgs[0])

plt.imshow(img)
plt.show()


import tensorflow as tf

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
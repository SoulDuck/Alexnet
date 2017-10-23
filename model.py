import cnn
import tensorflow
import numpy as np
import tensorflow as tf

def dropout( _input , is_training , keep_prob=0.8):
    if keep_prob < 1:
        output = tf.cond(is_training, lambda: tf.nn.dropout(_input, keep_prob), lambda: _input)
    else:
        output = _input
    return output


def batch_norm( _input , is_training):
    output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                          is_training=is_training, updates_collections=None)
    return output

def weight_variable_msra(shape , name):
    return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.variance_scaling_initializer())
def weight_variable_xavier( shape , name):
    return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape  , name='bias' ):
    initial=tf.constant(0.0 , shape=shape)
    return tf.get_variable(name,initializer=initial)
def conv2d_with_bias(_input , out_feature , kernel_size , strides , padding):
    in_feature=int(_input.get_shape()[-1])
    kernel=weight_variable_msra([kernel_size,kernel_size,in_feature, out_feature] , name='kernel')
    layer=tf.nn.conv2d(_input, kernel, strides, padding) + bias_variable(shape=[out_feature])
    print layer
    return layer

def fc_with_bias(_input , out_features ):
    in_fearues=int(_input.get_shape()[-1])
    kernel=weight_variable_xavier([in_fearues , out_features] , name='kernel')
    layer =tf.matmul(_input, kernel) + bias_variable(shape=[out_features])
    print layer
    return layer

def avg_pool( _input , k ):
    ksize=[1,k,k,1]
    strides=[1,k,k,1]
    padding='VALID'
    output=tf.nn.avg_pool(_input , ksize ,strides,padding)
    return output

def affine(name,x,out_ch ,keep_prob):
    with tf.variable_scope(name) as scope:
        if len(x.get_shape())==4:
            batch, height , width , in_ch=x.get_shape().as_list()
            w_fc=tf.get_variable('w' , [height*width*in_ch ,out_ch] , initializer= tf.contrib.layers.xavier_initializer())
            x = tf.reshape(x, (-1, height * width * in_ch))
        elif len(x.get_shape())==2:
            batch, in_ch = x.get_shape().as_list()
            w_fc=tf.get_variable('w' ,[in_ch ,out_ch] ,initializer=tf.contrib.layers.xavier_initializer())
        b_fc=tf.Variable(tf.constant(0.1 ), out_ch)
        layer=tf.matmul(x , w_fc) + b_fc
        layer=tf.nn.relu(layer)
        layer=tf.nn.dropout(layer , keep_prob)

        print 'layer name :'
        print 'layer shape :',layer.get_shape()
        print 'layer dropout rate :',keep_prob
        return layer


def fc_layer(_input ,out_feature , act_func='relu' , dropout='True' ):
    assert len(_input.get_shape()) == 2 , len(_input.get_shape())
    in_features=_input.get_shape()[-1]
    w = weight_variable_xavier([in_features, out_feature], name='W')
    b = bias_variable(shape=out_feature)
    layer= tf.matmul(_input, w) + b
    if act_func =='relu':
        layer=tf.nn.relu(layer)
    return layer





    return output
def fc_layer_to_clssses(_input , n_classes , is_training):
    output = batch_norm(_input , is_training=is_training)
    output = tf.nn.relu(output)
    last_pool_kernel = int(output.get_shape()[-2])
    output=avg_pool(output , k=last_pool_kernel)
    features_total=int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    W=weight_variable_xavier([features_total , n_classes] , name ='W')
    bias = bias_variable([n_classes])
    logits=tf.matmul(output, W)+bias
    return logits


def build_graph(x_ , y_ , is_training , conv_keep_prob , fc_keep_prob):
    ##### define conv connected layer #######
    n_classes=int(y_.get_shape()[-1])

    conv_out_features=[16,32,64,128,256]
    conv_kernel_sizes=[5,5,5,3,3]
    conv_strides=[2,2,2,2,2]
    before_act_bn_mode = [0, 1, 2, 3, 4]
    after_act_bn_mode = []


    allow_max_pool_indices=[0,1,4]
    conv_keep_prob=0.8

    assert len(conv_out_features) == len(conv_kernel_sizes )== len(conv_strides)
    layer=x_
    for i in range(len(conv_out_features)):
        with tf.variable_scope('conv_{}'.format(str(i))) as scope:
            if i in before_act_bn_mode:
                layer=batch_norm(layer , is_training)
            layer  = conv2d_with_bias(layer, conv_out_features[i], kernel_size=conv_kernel_sizes[i], \
                                 strides= [ 1, conv_strides[i], conv_strides[i], 1 ], padding='SAME' )
            if i in allow_max_pool_indices:
                layer=tf.nn.max_pool(layer , ksize=[1,2,2,1] , strides=[1,2,2,1] , padding='SAME')
                print layer
            layer = tf.nn.relu(layer)
            if i in after_act_bn_mode:
                layer = batch_norm(layer, is_training)

            layer=tf.nn.dropout(layer , keep_prob=conv_keep_prob)
    end_conv_layer=layer
    layer = tf.contrib.layers.flatten(end_conv_layer)
    ##### define fully connected layer #######
    fc_out_features = [1024,1024, n_classes]
    fc_keep_prob = 0.5


    before_act_bn_mode = [0, 1,]
    after_act_bn_mode = []
    for i in range(len(fc_out_features)):
        with tf.variable_scope('fc_{}'.format(str(i))) as scope:
            if i in before_act_bn_mode:
                batch_norm(layer , is_training)
            layer=fc_with_bias(layer , fc_out_features[i] )
            layer=tf.nn.relu(layer)
            layer = tf.nn.dropout(layer, keep_prob=fc_keep_prob ,)
            if i in after_act_bn_mode:
                batch_norm(layer, is_training)
    return layer




def train_algorithm_momentum(logits, labels, learning_rate):
    prediction = tf.nn.softmax(logits, name='softmax')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                   name='cross_entropy')
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()], name='l2_loss')
    momentum = 0.9;
    weight_decay = 1e-4
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=momentum, use_nesterov=True)
    train_op = optimizer.minimize(cross_entropy + l2_loss * weight_decay, name='train_op')
    correct_prediction = tf.equal(
        tf.argmax(prediction, 1),
        tf.argmax(labels, 1), name='correct_prediction')

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name='accuracy')
    return train_op, accuracy, cross_entropy, prediction


def define_inputs(shape, n_classes):
    images = tf.placeholder(
        tf.float32,
        shape=shape,
        name='x_')

    labels = tf.placeholder(
        tf.float32,
        shape=[None, n_classes],
        name='y_')

    learning_rate = tf.placeholder(
        tf.float32,
        shape=[],
        name='learning_rate')
    is_training = tf.placeholder(tf.bool, shape=[])
    return images, labels, learning_rate, is_training

def sess_start(logs_path):
    saver=tf.train.Saver()
    sess=tf.Session()
    summary_writer = tf.summary.FileWriter(logs_path)
    summary_writer.add_graph(tf.get_default_graph())
    init = tf.group(tf.global_variables_initializer() , tf.local_variables_initializer())
    sess.run(init)
    return sess, saver , summary_writer


def write_acc_loss(summary_writer ,prefix , loss , acc  , step):
    summary = tf.Summary(value=[tf.Summary.Value(tag='loss_{}'.format(prefix), simple_value=float(loss)),
                                tf.Summary.Value(tag='accuracy_{}'.format(prefix), simple_value=float(acc))])
    summary_writer.add_summary(summary, step)

if __name__ == '__main__':
    x_ , y_ , lr_ , is_training =define_inputs(shape=[None , 299,299, 3 ] , n_classes=2 )
    build_graph(x_=x_ , y_=y_ ,is_training=False ,conv_keep_prob=0.8 , fc_keep_prob=0.5)


#-*- coding:utf-8 -*-
import model
import input
import os
import fundus
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_imgs = mnist.train.images.reshape([-1,28,28,1])
train_labs = mnist.train.labels
test_imgs = mnist.test.images.reshape([-1,28,28,1])
test_labs = mnist.test.labels
train_fnames=None
test_fnames=None

print np.shape(train_imgs)
print np.shape(train_labs)

print np.shape(test_imgs)
print np.shape(test_labs)

#normalize
if np.max(train_imgs) > 1:
    train_imgs=train_imgs/255.
    test_imgs=test_imgs/255.
    print 'train_imgs max :',np.max(train_imgs)
    print 'test_imgs max :', np.max(test_imgs)

h,w,ch=train_imgs.shape[1:]
n_classes=np.shape(train_labs)[-1]

x_ , y_ , lr_ , is_training = model.define_inputs(shape=[None, h ,w, ch ] , n_classes=n_classes )
logits=model.build_graph(x_=x_ , y_=y_ ,is_training=is_training)
train_op, accuracy_op , loss_op , pred_op = model.train_algorithm_momentum(logits=logits,  labels=y_ , learning_rate=lr_)
sess, saver , summary_writer =model.sess_start('./logs/mnist')

if not os.path.isdir('./models'):
    os.mkdir('./models')

max_iter=2000000
ckpt=100
batch_size=80
share=len(test_labs)/batch_size
for step in range(max_iter):
    if step % ckpt==0:
        """ #### testing ### """
        test_fetches = [ accuracy_op, loss_op, pred_op ]
        val_acc_mean , val_loss_mean , pred_all = [] , [] , []
        for i in range(share): #여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
            test_feedDict = { x_: test_imgs[i*batch_size:(i+1)*batch_size], y_: test_labs[i*batch_size:(i+1)*batch_size], lr_: 0.01, is_training: False }
            val_acc ,val_loss , pred = sess.run( fetches=test_fetches, feed_dict=test_feedDict )
            val_acc_mean.append(val_acc)
            val_loss_mean.append(val_loss)
            pred_all.append(pred)
        val_acc_mean=np.mean(np.asarray(val_acc_mean ))
        val_loss_mean=np.mean(np.asarray(val_loss_mean))
        print 'validation acc : {} loss : {}'.format( val_acc_mean, val_loss_mean )
        model.write_acc_loss( summary_writer, 'validation', loss=val_acc_mean, acc=val_loss_mean, step=step)
        saver.save(sess=sess,save_path='./models/mnist/model' , global_step=step)
    """ #### training ### """
    train_fetches = [train_op, accuracy_op, loss_op]
    batch_xs, batch_ys, batch_fs = input.next_batch(batch_size, train_imgs, train_labs, train_fnames)
    train_feedDict = {x_: batch_xs, y_: batch_ys, lr_: 0.01, is_training: True}
    _ , train_acc, train_loss = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
    #print 'train acc : {} loss : {}'.format(train_acc, train_loss)
    model.write_acc_loss(summary_writer ,'train' , loss= train_loss , acc=train_acc  ,step= step)




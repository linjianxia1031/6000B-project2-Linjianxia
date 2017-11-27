# -*- coding: utf-8 -*-
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

def read_img(path,width,high):
    flower_cat = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    # del flower_cat[1]
    imgs = []
    labels = []
    for index, folder in enumerate(flower_cat):
        print index
        print folder
#         if folder != './data/flower_photos/.ipynb_checkpoints':
        for pic in glob.glob(folder + '/*.jpg'):
            img = io.imread(pic)
            img = transform.resize(img,(width,high))
            # print img
            imgs.append(img)
            labels.append(index)
    print len(labels)
    return np.asarray(imgs,np.float),np.asarray(labels,np.int32)


# def mix_data(data,labels):


def add_cnn_layers(name,input,shape,stride):
    with tf.variable_scope(name):
        conv_weights = tf.get_variable('weight',shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv_biases = tf.get_variable('bias', [shape[3]], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, conv_weights, strides=stride, padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
    return relu

def add_pooling_layers(name,input):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        return pool



def ful_con_layers(name,input,input_size,output_size):
    with tf.variable_scope(name):
        weights = tf.get_variable('weight', [input_size, output_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        f = tf.nn.relu(tf.matmul(input, weights) + biases)
        # if train: f = tf.nn.dropout(f, 0.5)
        return f,weights

def ful_con_layer(name,input,input_size,output_size):
    with tf.variable_scope(name):
        weights = tf.get_variable('weight', [input_size, output_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        f = tf.matmul(input, weights) + biases
        # if train: f = tf.nn.dropout(f, 0.5)
        return f,weights



datapath = './data/flower_photos/'
# model_path = './data/models.ckpt'

width = 64
high = 64
color = 3
data, label = read_img(datapath,width,high)

num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

tst_path = '../data/test/'
tst_img = []
for pic in glob.glob(tst_path + '/*.jpg'):
    img = io.imread(pic)
    img = transform.resize(img,(width,high))
    # print img
    tst_img.append(img)
print len(tst_img)
tst_img = np.asarray(tst_img,np.float)





x=tf.placeholder(tf.float32,shape=[None,width,high,color],name='x')
y=tf.placeholder(tf.int32,shape=[None,],name='y')

relu1 = add_cnn_layers('conv1',x,[5,5,3,32],[1,1,1,1])
pool1 = add_pooling_layers('pool1',relu1)
relu2 = add_cnn_layers('conv2',pool1,[5,5,32,64],[1,1,1,1])
pool2 = add_pooling_layers('pool2',relu2)
relu3 = add_cnn_layers('conv3',pool2,[5,5,64,128],[1,1,1,1])
pool3 = add_pooling_layers('pool3',relu3)
relu4 = add_cnn_layers('conv4',pool3,[3,3,128,128],[1,1,1,1])
pool4 = add_pooling_layers('pool4',relu4)
relu5 = add_cnn_layers('conv5',pool4,[3,3,128,128],[1,1,1,1])
pool5 = add_pooling_layers('pool5',relu5)
nodes = 2*2*128
input_data = tf.reshape(pool5,[-1,nodes])
ful1,weights1 = ful_con_layers('ful1',input_data,nodes,1024)
ful2,weights2 = ful_con_layer('ful2',ful1,1024,5)
# ful3,weights3 = ful_con_layers('ful3',ful2,512,512)
# ful4,weights4 = ful_con_layer('ful4',ful3,512,5)
# ful3 = tf.nn.softmax(ful3)
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ful2, labels=y)+0.001*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2))
# loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ful2, labels=y)

train=tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
classes = tf.argmax(ful2,1)
correct_prediction = tf.equal(tf.cast(classes,tf.int32), y)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

n_epoch = 10
batch_size = 32
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train, loss, acc], feed_dict={x: x_train_a, y: y_train_a})
#         print len(sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a}))
        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
    print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
pre_tst = sess.run(classes,feed_dict={x: tst_img})
np.savetxt('test_label.csv',pre_tst, delimiter = ' ')
saver.save(sess, model_path)
sess.close()

# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess = tf.Session()
#     sess.run(init)
#     for i in range(0,100):
#         sess.run(train, feed_dict={x: x_train, y: y_train})
#         print sess.run(acc, feed_dict={x: x_train, y: y_train})
#         print sess.run(ful2, feed_dict={x: x_train, y: y_train})

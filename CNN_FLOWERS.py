# -*- coding: utf-8 -*-
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

def read_img(path,width,high):
    flower_cat = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    del flower_cat[1]
    imgs = []
    labels = []
    for index, folder in enumerate(flower_cat):
        print index
        print folder
        for pic in glob.glob(folder + '/*.jpg'):
            img = io.imread(pic)
            img = transform.resize(img,(width,high))
            imgs.append(img)
            labels.append(index)
    print len(labels)
    return np.asarray(imgs,np.float),np.asarray(labels,np.int32)


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
        return f,weights

def ful_con_layer(name,input,input_size,output_size):
    with tf.variable_scope(name):
        weights = tf.get_variable('weight', [input_size, output_size],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('bias', [output_size], initializer=tf.constant_initializer(0.1))
        f = tf.matmul(input, weights) + biases
        return f,weights



datapath = './data/flower_photos/'
model_path = './data/models.ckpt'

width = 224
high = 224
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
x=tf.placeholder(tf.float32,shape=[None,width,high,color],name='x')
y=tf.placeholder(tf.int32,shape=[None,],name='y')

#conv1
conv1_1 = add_cnn_layers('conv1.1',x,[3,3,3,64],[1,1,1,1])
conv1_2 = add_cnn_layers('conv1.2',conv1_1,[3,3,64,64],[1,1,1,1])
pool1 = add_pooling_layers('pool1',conv1_2)

#conv2
conv2_1 = add_cnn_layers('conv2.1',pool1,[3,3,64,128],[1,1,1,1])
conv2_2 = add_cnn_layers('conv2.2',conv2_1,[3,3,128,128],[1,1,1,1])
pool2 = add_pooling_layers('pool2',conv2_2)

#conv3
conv3_1 = add_cnn_layers('conv3.1',pool2,[3,3,128,256],[1,1,1,1])
conv3_2 = add_cnn_layers('conv3.2',conv3_1,[3,3,256,256],[1,1,1,1])
conv3_3 = add_cnn_layers('conv3.3',conv3_2,[3,3,256,256],[1,1,1,1])
pool3 = add_pooling_layers('pool3',conv3_3)

#conv4
conv4_1 = add_cnn_layers('conv4.1',pool3,[3,3,256,512],[1,1,1,1])
conv4_2 = add_cnn_layers('conv4.2',conv4_1,[3,3,512,512],[1,1,1,1])
conv4_3 = add_cnn_layers('conv4.3',conv4_2,[3,3,512,512],[1,1,1,1])
pool4 = add_pooling_layers('pool4',conv4_3)

#conv5
conv5_1 = add_cnn_layers('conv5.1',pool4,[3,3,512,512],[1,1,1,1])
conv5_2 = add_cnn_layers('conv5.2',conv5_1,[3,3,512,512],[1,1,1,1])
conv5_3 = add_cnn_layers('conv5.3',conv5_2,[3,3,512,512],[1,1,1,1])
pool5 = add_pooling_layers('pool5',conv5_3)


nodes = 7*7*512
input_data = tf.reshape(pool5,[-1,nodes])
ful1,weights1 = ful_con_layers('ful1',input_data,nodes,1024)
ful2,weights2 = ful_con_layers('ful2',ful1,1024,1024)
ful3,weights3 = ful_con_layer('ful3',ful2,1024,5)
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ful3, labels=y)+0.001*(tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2))


train=tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(ful2,1),tf.int32), y)
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
        train_loss += err;
        train_acc += ac;
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
saver.save(sess, model_path)
sess.close()


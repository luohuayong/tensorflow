# -*- coding:utf-8 -*-

"""
学习示例
- 测试mnist
- lenet-5 网络
- 从pb文件载入训练结果
- 测试正确率
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_PATH = "../data"
MNIST_DATA = DATA_PATH + "/MNIST_data"
PB_PATH = DATA_PATH + "/model/mnist.pb"
mnist = input_data.read_data_sets(MNIST_DATA, one_hot=True)

sess = tf.InteractiveSession()
# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial, name=name)
#
#
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

graph_def = tf.GraphDef()
# 从pb文件载入训练结果
with tf.gfile.GFile(PB_PATH, "rb") as f:
    graph_def.ParseFromString(f.read())
W_conv1 = tf.import_graph_def(graph_def, return_elements=["W_conv1:0"])
W_conv1 = tf.reshape(W_conv1, [5, 5, 1, 32])
b_conv1 = tf.import_graph_def(graph_def, return_elements=["b_conv1:0"])
b_conv1 = tf.reshape(b_conv1, [32])
W_conv2 = tf.import_graph_def(graph_def, return_elements=["W_conv2:0"])
W_conv2 = tf.reshape(W_conv2, [5, 5, 32, 64])
b_conv2 = tf.import_graph_def(graph_def, return_elements=["b_conv2:0"])
b_conv2 = tf.reshape(b_conv2, [64])
W_fc1 = tf.import_graph_def(graph_def, return_elements=["W_fc1:0"])
W_fc1 = tf.reshape(W_fc1, [7 * 7 * 64, 1024])
b_fc1 = tf.import_graph_def(graph_def, return_elements=["b_fc1:0"])
b_fc1 = tf.reshape(b_fc1, [1024])
W_fc2 = tf.import_graph_def(graph_def, return_elements=["W_fc2:0"])
W_fc2 = tf.reshape(W_fc2, [1024, 10])
b_fc2 = tf.import_graph_def(graph_def, return_elements=["b_fc2:0"])
b_fc2 = tf.reshape(b_fc2, [10])

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("test accuracy %g" % sess.run(accuracy, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

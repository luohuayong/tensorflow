# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
包含BP的简单神经网络实现(2层 4个神经元)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from numpy.random import RandomState

dataset_size = 128
batch = 8
rate = 0.001
setps = 5000

# 生成训练数据集
X = RandomState(1).rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

W1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='y')

a = tf.matmul(x, W1)
y_ = tf.matmul(a, W2)

loss = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_, 1e-10, 1.0)))
train = tf.train.AdamOptimizer(rate).minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("begin W1 = %s" % sess.run(W1))
    print("begin W2 = %s" % sess.run(W2))
    for i in range(setps):
        start = (i*batch) % dataset_size
        end = min(start + batch, dataset_size)
        sess.run(train, feed_dict={x: X[start:end], y: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y: Y})
            print("i=%s, total_loss=%s" % (i, total_loss))
    print("end W1 = %s" % sess.run(W1))
    print("end W2 = %s" % sess.run(W2))


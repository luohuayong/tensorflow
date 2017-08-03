# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
简单的神经网络实现(2层 4个神经元 无BP)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

W1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
W2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.constant([[0.7, 0.9]])
a = tf.matmul(x, W1)
y = tf.matmul(a, W2)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(y))


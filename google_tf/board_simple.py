# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
tensorboard 使用示例
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

LOG_PATH = "../data/log"
input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter(LOG_PATH, tf.get_default_graph())
writer.close()









# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
持久化代码实现
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def dump():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    init_op = tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, "../model/model.ckpt")


def load():
    saver = tf.train.import_meta_graph("../model/model.ckpt.meta")
    with tf.Session() as sess:
        saver.restore(sess, "../model/model.ckpt")

        print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


def save_json():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    saver = tf.train.Saver()
    saver.export_meta_graph("../model/model.ckpt.meta.json", as_text=True)


def read_checkpoint():
    reader = tf.train.NewCheckpointReader("../model/model.ckpt")
    all_variables = reader.get_variable_to_shape_map()
    for variable_name in all_variables:
        print(variable_name, all_variables[variable_name])
    print("v1 is :", reader.get_tensor("v1"))
    print("v2 is :", reader.get_tensor("v2"))

if __name__ == "__main__":
    # load()
    # save_json()
    read_checkpoint()


# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
持久化代码实现
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util

BASE_PATH = "../data"


def save_ckpt():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = tf.add(v1, v2, name="result")
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        print("v1:", sess.run(v1))
        print("v2:", sess.run(v2))
        print("result:", sess.run(result))
        saver.save(sess, BASE_PATH + "/model/model.ckpt")


def load_ckpt():
    saver = tf.train.import_meta_graph(BASE_PATH + "/model/model.ckpt.meta")
    graph_def = tf.get_default_graph()
    with tf.Session() as sess:
        saver.restore(sess, BASE_PATH + "/model/model.ckpt")
        print("v1:", sess.run(graph_def.get_tensor_by_name("v1:0")))
        print("v2:", sess.run(graph_def.get_tensor_by_name("v2:0")))
        print("result:", sess.run(graph_def.get_tensor_by_name("result:0")))


def save_pb():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = tf.add(v1, v2, name="result")
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(result)
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name, op.values())
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, ['result'])
        with tf.gfile.GFile(BASE_PATH + "/model/model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())


def load_pb():
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(BASE_PATH + "/model/model.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
    result = tf.import_graph_def(graph_def, return_elements=["result:0"])
    with tf.Session() as sess:
        print("result:", sess.run(result))
        # for op in tf.get_default_graph().get_operations():
        #     print(op.name, op.values())



def save_json():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2
    saver = tf.train.Saver()
    saver.export_meta_graph(BASE_PATH + "/model/model.ckpt.meta.json", as_text=True)


def read_checkpoint():
    reader = tf.train.NewCheckpointReader(BASE_PATH + "/model/model.ckpt")
    all_variables = reader.get_variable_to_shape_map()
    for variable_name in all_variables:
        print(variable_name, all_variables[variable_name])
    print("v1 is :", reader.get_tensor("v1"))
    print("v2 is :", reader.get_tensor("v2"))

if __name__ == "__main__":
    # save_ckpt()
    # load_ckpt()
    save_pb()
    load_pb()
    # save_json()
    # read_checkpoint()


# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
MNIST训练神经网络最佳实践，包含三个文件
- mnist_inference 前向传播
- mnist_train 训练过程
- mnist_eval 测试过程
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [LAYER1_NODE],
            initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable(
            "biases", [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2



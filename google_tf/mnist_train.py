# -*- coding:utf-8 -*-

"""
Tensorflow 实战google深度学习框架示例程序
简单的多线程示例
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import threading
import time

def myloop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("stoping from id: %d" % worker_id)
            coord.request_stop()
        else:
            print("working on id: %d" % worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()
threads = [threading.Thread(target=myloop, args=(coord, i)) for i in range(5)]
for t in threads:
    t.start()
coord.join(threads)
print("all thread stoped !")











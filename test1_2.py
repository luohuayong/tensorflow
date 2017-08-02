from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)



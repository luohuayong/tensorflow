from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
### create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
### create tensorflow structure end
sess = tf.Session()
sess.run(init)
for item in range(201):
    sess.run(train)
    if item % 20 == 0 :
        print(item,sess.run(Weights),sess.run(biases))


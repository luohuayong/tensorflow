import tensorflow as tf
import tensorflow.examples.tutorials.mnist

mnist = tf.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/",one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.nn.softmax(tf.matmul(x,W)+b)
cross = -tf.reduce_sum(y*tf.log(y_))
train = tf.train.GradientDescentOptimizer(0.01).minimize(cross)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch_x,batch_y = mnist.train.next_batch(100)
    sess.run(train,feed_dict={x:batch_x,y:batch_y})
    correct = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    if i % 20 == 0:
        print sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})

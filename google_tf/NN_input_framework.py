# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

files = tf.train.match_filenames_once("../data/flower_photos/*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'lable': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
    })
image = features['image']
label = features['lable']
height = features['height']
width = features['width']
channels = features['channels']
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

image_size = 299
distorted_image = preprocess_for_train(
    decoded_image, image_size, image_size, None
)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label], batch_size=batch_size,
    capacity=capacity, min_after_dequeue=min_after_dequeue)

logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learing_rate)
.minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(TRAINING_ROUDNS):
        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)

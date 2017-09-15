# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 13:37:58 2017

@author: NepTuNe
"""

import numpy as np
import tensorflow as tf
import model

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement = True, device_count = {'GPU': 1})
    sess = tf.InteractiveSession(config = config)
    #saver = tf.train.Saver()
    x = tf.placeholder(tf.float32, shape = [None, 44,44])
    y_ = tf.placeholder(tf.float32, shape = [None, 2])


    
    x_image = tf.reshape(x, [-1,44,44,1])
    layer1 = model.add_layer(x_image, model.weights_variable([5, 5, 1, 32], "w_conv1"), model.bias_variable([32], "b_conv1"))


    layer2 = model.add_layer(layer1, model.weights_variable([5, 5, 32, 64], "w_conv2"), model.bias_variable([64], "b_conv2"))

    W_fc1 = model.weights_variable([11 *11 * 64, 1024], "w_fc1")
    b_fc1 = model.bias_variable([1024], "b_fc1")
    h_pool2_flat = tf.reshape(layer2, [-1, 11 * 11 * 64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = model.weights_variable([1024, 2], "w_fc2")
    b_fc2 = model.bias_variable([2], "b_fc2")
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    batch_size = 50
    image_batch, label_batch = model.input_pipeline('train.tfrecords', batch_size)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print ('Start Training!')
    for i in range(20000):    
        
        images,labels = sess.run([image_batch, label_batch])
        if i % 100 == 0:
            train_accuracy = accuracy.eval( feed_dict = { x: images, y_: labels, keep_prob : 1.0})
            print('step %d training accuracy%g '%(i, train_accuracy))
        train_step.run(feed_dict = { x: images, y_: labels, keep_prob: 0.5})
        
    coord.request_stop()
    coord.join(threads)
    #save_path = saver.Save(sess, path)





    


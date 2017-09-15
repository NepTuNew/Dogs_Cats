#model for cnn network
import tensorflow as tf
import numpy as np
from skimage import io
from skimage import transform

import random
import os


def weights_variable(shape, name = 'None'):
    initial = tf.truncated_normal(shape=shape, stddev = 0.1)
    return tf.Variable(initial, name = name)
    
def bias_variable(shape, name = 'None'):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial, name = name)

def conv2d(X,W):
    return tf.nn.conv2d(X,W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(X):
    return tf.nn.max_pool(X, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
def add_layer(X,W,B):
    h_conv = tf.nn.relu(conv2d(X,W) + B)
    return max_pool_2x2(h_conv)

def next_batch( total, num):    
    choose = randomChoose(total,num)
    datas = np.zeros([num, 44*44*1])
    labels = np.zeros([num, 2])
    for index, i in enumerate(choose):
        if i < int(total/2):
            filename = os.path.join('C:/Users/NepTuNe/Desktop/Dogs_Cats/train/' + 'cat.' + str(i) + '.jpg')
            #print(filename)
            labels[index,0] = 1
        else:
            filename = os.path.join('C:/Users/NepTuNe/Desktop/Dogs_Cats/train/' + 'dog.' + str(i-12500) + '.jpg')
            #print(filename)
            labels[index,1] = 1
        image = io.imread(filename, as_grey = True)
        resizeImage = transform.resize(image, [44,44])        
        datas[index,:] = resizeImage.reshape([44*44])
    return (datas, labels)
def randomChoose(total, num):
    base = 12500
    select_list1 = range(0,int(total/2))
    select_list2 = range(base,int(base+total/2))
    choose1 = random.sample(select_list1, int(num/2))
    choose2 = random.sample(select_list2, int(num/2))
    output = choose1 + choose2
    return output
    
def get_test_data( num ): # used for self testing
    datas = np.zeros([num, 44*44])
    labels = np.zeros([num,2])
    base = 10500
    select_list = range(0,4000)
    choose = random.sample(select_list, num)
    for index, i in enumerate(choose):
        if i < 2000:
            filename = os.path.join('C:/Users/NepTuNe/Desktop/Dogs_Cats/train/0/' + 'cat.' + str(i+base) + '.jpg')
            labels[index, 0] = 1
        else :
            filename = os.path.join('C:/Users/NepTuNe/Desktop/Dogs_Cats/train/1/' + 'dog.' + str(i+base-2000) + '.jpg')
            labels[index, 1] = 1
        image = io.imread(filename, as_grey = True)
        resizeImage = transform.resize(image, [44,44])
        datas[index, :] = resizeImage.reshape([44*44])
    return (datas, labels)
    
def read_and_decode(filename):

    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [44, 44])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(indices = label, depth = 2, on_value = 1, off_value = 0)
    return img, label

def input_pipeline(filename, batch_size):
    images, labels = read_and_decode(filename)
    min_after_dequeue = 25000
    capacity = min_after_dequeue + 3*batch_size
    example_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity = capacity, min_after_dequeue = min_after_dequeue)
    return example_batch, label_batch
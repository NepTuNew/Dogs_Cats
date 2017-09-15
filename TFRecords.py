# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:04:44 2017

@author: NepTuNe
"""
import os
import tensorflow as tf
from skimage import io
from skimage import transform


def make_TFRecords(cwd = 'C:/Users/NepTuNe/Desktop/Dogs_Cats/train', output_name = 'train.TFRecords'):
        
    writer = tf.python_io.TFRecordWriter(output_name)
    for classes in os.listdir(cwd): #I have 0 named and 1 named folder 0 for cats and 1 for dogs
        class_path = cwd + '/' + classes + '/'
        for img_name in os.listdir(class_path): # each class has 12500 images
            img_path = class_path + img_name
            img = io.imread(img_path, as_grey = True)
            img = transform.resize(img, [44,44])
            img_raw = img.tobytes()
            example = tf.train.Example(features = tf.train.Features(feature={'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[int(classes)])), 'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
            writer.write(example.SerializeToString())
    writer.close()
    
if __name__ ==  '__main__':

    print('Start to make TFRecords')
    make_TFRecords()    
    print('Done!')
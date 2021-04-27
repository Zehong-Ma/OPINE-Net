'''
@File    :   load_tfrecord_sample.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   load data from tfrecord, and test whether convertion from mat to tfrecord is successful 
'''

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def _parse_record(example_photo):
    features = {
        'label': tf.FixedLenFeature([1089], tf.float32)
    }
    parsed_features = tf.parse_single_example(example_photo,features=features)
    return parsed_features['label']

dataset = tf.data.TFRecordDataset('Training_Data.tfrecord')
dataset = dataset.map(_parse_record)
dataset = dataset.repeat().shuffle(1000).batch(4)
iterator = dataset.make_one_shot_iterator()
element = iterator.get_next()
temp = 2* element
with tf.Session() as sess:
    out = sess.run(temp)
    print(out)
    print(out.shape)
'''
@File    :   covert_mat_to_tfrecord.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   convert mat to tfrecord
'''

import scipy.io as sio
import tensorflow as tf
Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat(Training_data_Name)
Training_labels = Training_data['labels']

filename =  ('Training_Data.tfrecord')  
n_samples = len(Training_labels)
writer = tf.python_io.TFRecordWriter(filename)  
print('\nTransform start......')  
for i in range(n_samples):
    example = tf.train.Example(
                    features=tf.train.Features(feature={  
                    'label':tf.train.Feature(float_list=tf.train.FloatList(value=Training_labels[i]))
                    }))  
    writer.write(example.SerializeToString())  
    if i%100==0:
        print(i)

writer.close()  
print('Transform done!')  

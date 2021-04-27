'''
@File    :   test_model.py
@Author  :   Zehong Ma
@Version :   1.0
@Contact :   zehongma@qq.com
@Desc    :   restore the OPINE-Net(plus) and test on the Set11
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cv2
import glob
import os
from utils import *
from modules import *
from skimage.measure import compare_ssim as ssim
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

def load_imgs_to_array():
    test_dir = os.path.join(FLAGS.data_dir, FLAGS.test_name)
    filepaths = glob.glob(test_dir + '/*.tif')
    result_dir = os.path.join(FLAGS.result_dir, FLAGS.test_name)

    ImgNum = len(filepaths)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


def evaluate():
    
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ####################
        # Define the model #
        ####################
        image_width = (FLAGS.test_image_width//33+1)*33
        image_height = image_width
        batch_x = tf.placeholder(dtype=tf.float32,shape=(1,image_height,image_width,1))
      
        Phix, Phi_weight, Phi = sampling_subnet(batch_x)
        x_0, Phi_T_weight = initialization_subnet(Phix, Phi)
        x_final, layers_sym = recovery_subnet(x_0, Phi_weight, Phi_T_weight)

        variables_to_restore = tf.trainable_variables()
        saver = tf.train.Saver(variables_to_restore)
        
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    if os.path.isabs(ckpt.model_checkpoint_path):
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    else:
                        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                        saver.restore(sess, os.path.join(FLAGS.model_dir, ckpt_name))
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    print('Succesfully loaded model from %s at step=%s.' %
                        (ckpt.model_checkpoint_path, global_step))
                else:
                    print('No checkpoint file found')
                    return

                test_dir = os.path.join(FLAGS.data_dir, FLAGS.test_name)
                filepaths = glob.glob(test_dir + '/*.tif')
                result_dir = os.path.join(FLAGS.result_dir, FLAGS.test_name)

                ImgNum = len(filepaths)
                PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
                SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

                print('\n')
                print("CS Sampling and Reconstruction by OPINE-Net plus Start")
                print('\n')
                for img_no in range(ImgNum):

                    imgName = filepaths[img_no]

                    Img = cv2.imread(imgName, 1)
                    if Img.shape[1]!=FLAGS.test_image_width:
                        continue
                        # Img = cv2.resize(Img,(256,256))

                    Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()

                    Iorg_y = Img_yuv[:,:,0]

                    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)

                    Img_output = Ipad.reshape(1, row_new, col_new, 1)/255.0

                    start = time()

                    x_output = sess.run([x_final],feed_dict={batch_x:Img_output})


                    end = time()

                    Prediction_value = np.squeeze(x_output)

                    X_rec = np.clip(Prediction_value[:row,:col], 0, 1).astype(np.float64)

                    rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
                    rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)

                    print("[%02d/%02d] Run time for %s is %.4f, PSNR is %.2f, SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

                    Img_rec_yuv[:,:,0] = X_rec*255

                    im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                    im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)

                    resultName = imgName.replace(FLAGS.data_dir, FLAGS.result_dir)
                    print(resultName)
                    cv2.imwrite("%s_OPINE_Net_plus_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, FLAGS.cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)
                    del x_output

                    PSNR_All[0, img_no] = rec_PSNR
                    SSIM_All[0, img_no] = rec_SSIM
                print('\n')
                output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f \n" % (FLAGS.cs_ratio, FLAGS.test_name, np.mean(PSNR_All), np.mean(SSIM_All))
                print(output_data)

                output_file_name = "./%s/PSNR_SSIM_Results_CS_OPINE_Net_plus_layer_%d_group_%d_ratio_%d.txt" % (FLAGS.log_dir, FLAGS.layer_num, FLAGS.group_num, FLAGS.cs_ratio)

                output_file = open(output_file_name, 'a')
                output_file.write(output_data)
                output_file.close()

                print("CS Sampling and Reconstruction by OPINE-Net plus End")
#coding=utf-8
import numpy as np
from scipy import misc # feel free to use another image loader
import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/home/z/PycharmProjects/pre_bias_cnn/model_saved/model.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/z/PycharmProjects/pre_bias_cnn/model_saved/'))


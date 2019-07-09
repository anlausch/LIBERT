# https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow
from debug_util import *
import tensorflow as tf
k = tf.constant([
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
], dtype=tf.float32, name='k')
i = tf.constant([
    [4, 3, 1, 0],
    [2, 1, 0, 1],
    [1, 2, 4, 1],
    [3, 1, 0, 2]
], dtype=tf.float32, name='i')
kernel = tf.reshape(k, [3, 3, 1, 1], name='kernel')
image  = tf.reshape(i, [1, 4, 4, 1], name='image')
res_before_squeeze = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID")
res2_before_squeeze = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "SAME")

res = tf.squeeze(res_before_squeeze)
res2 = tf.squeeze(res2_before_squeeze)
pass
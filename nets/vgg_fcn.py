#!/usr/bin/env python
#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from config.config import FLAGS
slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.INFO)

def vgg_arg_scope(weight_decay=0.0005):
   """Defines the VGG arg scope.
   
   Args:
     weight_decay: The l2 regularization coefficient.
   
   Returns:
     An arg_scope.
   """
   with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                       activation_fn=tf.nn.relu,
                       weights_regularizer=slim.l2_regularizer(weight_decay),
                       biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
         return arg_sc
     
def vgg_fcn(images,
            num_classes=FLAGS.num_classes,
            is_training=True,
            dropout_keep_prob=0.5,
            spatial_squeeze=False, #True,
            scope='vgg_16',
            fc_conv_padding='VALID'):
 
   """
    define slim form vgg
   """
   with tf.variable_scope(scope, 'vgg_16', [images]) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
         conv1 = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
         pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
         conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
         pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
         conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
         pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
         conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
         pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
         conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
         pool5 = slim.max_pool2d(conv5, [2, 2], scope='pool5')
         fc6 = slim.conv2d(pool5, 4096, [7, 7], stride=1, scope='fc6')
         drop6 = slim.dropout(fc6, dropout_keep_prob, is_training=is_training, scope='dropout6')
         fc7 = slim.conv2d(drop6, 4096, [1, 1], scope='fc7')
         drop7 = slim.dropout(fc7, dropout_keep_prob, is_training=is_training, scope='dropout7')
         fc8 = slim.conv2d(drop7, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
         #end_points = slim.utils.convert_collection_to_dict(end_points_collection)

         if spatial_squeeze:
            fc8 = tf.squeeze(fc8, [1, 2], name='fc8/squeezed')
         msk_shape = images.shape
         # it is intialized as bilinar interpolation(calculation predfined)
         score_fr = slim.conv2d(fc7, num_classes, [1, 1], scope='score_fr')
         tf.logging.info("score_fr size: %s"% tf.shape(score_fr))
         fc32s = slim.conv2d_transpose(inputs=score_fr, num_outputs=num_classes, kernel_size=[64, 64], stride=32, scope='fc32s')
         tf.logging.info("fc32s: %s" % tf.shape(fc32s))

         #fc32s = tf.image.crop_to_bounding_box(fc32s, 19, 19, msk_shape[1], msk_shape[2])
       
         upscore2 = slim.conv2d_transpose(inputs=score_fr, num_outputs=num_classes, kernel_size=[4, 4], stride=2, scope='upscore2')

         deconv_shape1 = upscore2.get_shape() 
         score_pool4 = slim.conv2d(pool4, num_classes, [1, 1], stride=1, scope='score_pool4')

         #score_pool4 = tf.image.crop_to_bounding_box(score_pool4, 5, 5, deconv_shape1[1], deconv_shape1[2])
 
         fuse_pool4 = tf.add(upscore2, score_pool4)

         fc16s = slim.conv2d_transpose(inputs=fuse_pool4, num_outputs=num_classes, kernel_size=[32, 32], stride=16, scope='fc16s')

         tf.logging.info("fc16s size: %s" % tf.shape(fc16s))
         #fc16s = tf.image.crop_to_bounding_box(fc16s, 27, 27, msk_shape[1], msk_shape[2])
         
         upscore_pool4 = slim.conv2d_transpose(inputs=fuse_pool4, num_outputs=num_classes, kernel_size=[4, 4], stride=2, scope='upscore_pool4')

         upscore_pool3 = slim.conv2d(pool3, num_classes, [1, 1], stride=1, scope='upscore_pool3')
         deconv_shape2 = upscore_pool4.get_shape()
         #upscore_pool3 = tf.image.crop_to_bounding_box(upscore_pool3, 9, 9, deconv_shape2[1], deconv_shape[2])
         fuse_pool3 = tf.add(upscore_pool4, upscore_pool3)

         fc8s = slim.conv2d_transpose(inputs=fuse_pool3, num_outputs=num_classes, kernel_size=[16, 16], stride=8, scope='fc8s')
         tf.logging.info("fc8s size: %s" % tf.shape(fc8s))
         #fc8s = tf.image.crop_to_bounding_box(fc8s, 31, 31, msk_shape[1], msk_shape[2])


         annotation_pred = tf.argmax(fc8s, dimension=3, name="prediction")
         end_points = slim.utils.convert_collection_to_dict(end_points_collection)

         tf.get_variable_scope().reuse_variables()

         return tf.expand_dims(annotation_pred, dim=3), fc8s, end_points

#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import time
import yaml
import random
import functools
import collections

import tensorflow as tf
from config.config import FLAGS
import numpy as np
from nets.vgg_fcn import vgg_arg_scope
from nets.vgg_fcn import vgg_fcn

import datasets.voc_preprocessing as _preprocessing

import utils.utils as utils

from PIL import Image
from PIL import ImageColor

slim = tf.contrib.slim

def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
  func = vgg_fcn
  @functools.wraps(func)
  def network_fn(images):
    arg_scope = vgg_arg_scope(weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
      return func(images, num_classes, is_training=is_training)
  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn 

def next_batch(files, batch_num):

    masks = []
    imgs = []
    labels = [] #preprocessed
    inputs = [] #preprocessed

    for filename in files[batch_num*FLAGS.batch_size: min((batch_num+1)*FLAGS.batch_size, len(files))]:
        im = Image.open(filename).convert('RGB')
        imgs.append(im) #Original image without resize
        #print(filename.split('/')[-1].split('.'))
        msk_file = os.path.join(FLAGS.dataset_dir, 'SegmentationClassAug', filename.split('/')[-1].split('.')[0]+'.png')

        mask = Image.open(msk_file)#.convert('L')
        masks.append(mask) #Ground truth segmentation results
        #print(im.size(), masks.size())
        val_image, val_label, _, _ = _preprocessing.preprocess_image(im, mask, None)
        inputs.append(val_image)
        labels.append(val_label) 

    val_images = np.array(inputs)
    val_images = np.squeeze(val_images)
    val_images = np.cast['float32'](val_images)
    val_labels = np.array(labels)
    val_labels = np.squeeze(val_labels)
    val_labels = np.cast['float32'](val_labels)
    return imgs, masks, val_images, val_labels

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def show_mask(image, mask, alpha=0.5):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    labels = np.unique(mask)
    labels = np.delete(labels, np.where(labels == 0))
    labels = np.delete(labels, np.where(labels == 255))
    print(labels)
    _image = Image.fromarray(image)
    for label in labels:
        index = np.where(mask == label)
        _mask = np.zeros(mask.shape, dtype=np.uint8)
        _mask[index] = 255*alpha
        #print(label.dtype, type(_mask), _mask.shape, _mask.dtype)
        _mask_im = Image.fromarray(_mask.astype(np.uint8), mode='L')
        _mask_im = _mask_im.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
        _mask = np.array(_mask_im)
        solid_color = np.expand_dims(np.ones_like(_mask), axis=2) * np.reshape(list(ImageColor.getrgb(STANDARD_COLORS[label])), [1,1,3])
        solid_color_im = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        _image = Image.composite(solid_color_im, _image, _mask_im)
        #for i in range(3):
        #    # image[index[0], index[1], i] = 0.5 * image[index[0], index[1], i] + #0.5 * GLcolor_map[label] #
        #    image[index[0], index[1], i] = 0.75 * image[index[0], index[1], i] - random.randint(17, 50) * 0.5

    return _image.convert('RGB') #Image.fromarray(_image.astype(np.uint8), mode='RGB')

# load model and graph
# load images from filelist

# predict via inference

# save results

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    sess = tf.Session()

    graph = tf.Graph().as_default()
    network_fn = get_network_fn(FLAGS.num_classes, is_training=False)
    images = tf.placeholder(tf.float32,(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3), name='images')
    labels = tf.placeholder(tf.float32,(FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size), name='labels')
    ex_labels = tf.expand_dims(labels, axis=[3])

    annotations, fc8s, end_points = network_fn(images)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'mean_iou': slim.metrics.streaming_mean_iou(annotations, ex_labels, 21),
      'accuracy': slim.metrics.streaming_accuracy(annotations, ex_labels),
    })
 
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    summaries.add(tf.summary.image("Original_images", images))                    
    summaries.add(tf.summary.image("Ground_truth_masks", ex_labels))
    summaries.add(tf.summary.image("Prediction_masks", tf.to_float(annotations)))


    summary_op = tf.summary.merge(list(summaries), name='summary_op')


    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):                         
       checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
       checkpoint_path = FLAGS.checkpoint_path
    
    tf.logging.info('Evaluating %s' % checkpoint_path)
    
    variables_to_restore = slim.get_variables_to_restore()
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    
    saver.restore(sess, checkpoint_path)
    
    files = list(map(lambda x: os.path.join(FLAGS.dataset_dir, 'JPEGImages', x.strip()+'.jpg'), open('val.txt')))
    #list(map(lambda x: os.path.join(FLAGS.dataset_dir, 'JPEGImages', x), os.listdir(os.path.join(FLAGS.dataset_dir, 'JPEGImages'))))

    num_batches = len(files)/FLAGS.batch_size


    for i in range(num_batches):
        print("Batch %d..." % i)
        original_images, gt_masks, val_images, val_labels = next_batch(files, i)
        #print(type(val_images), val_images.dtype, val_labels.dtype)
        #print(val_images.shape, val_labels.shape)
        feed_dict = {images:val_images, labels:val_labels}
        annotation, _, _ = sess.run([annotations, fc8s, end_points], feed_dict) #{val_images:val_images, val_labels:val_labels})


        for j in range(val_images.shape[0]):

            np_im = np.cast['uint8'](np.array(original_images[j]))
            np_mask = np.cast['uint8'](np.array(gt_masks[j]))
            gt_im_mask = show_mask(np_im, np_mask)
            gt_im_mask.save(os.path.join(FLAGS.logs_dir, 'results', 'mask%d_%4d.jpg' %(i,j)), 'JPEG')

            val_res = np.cast['uint8'](val_labels[j,:, :])
            masked_img = show_mask(np_im, val_res)
            masked_img.save(os.path.join(FLAGS.logs_dir, 'results', '%3d_%4d.jpg' %(i,j)), 'JPEG')
  

if __name__ == '__main__':
    main()

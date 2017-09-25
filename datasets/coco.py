from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

_FILE_PATTERN = 'coco_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train2014': 82783, 'val2014': 40504}

_NUM_CLASSES = 81

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'An annotation image of varying size. (pixel-level masks)',
    'gt_masks': 'masks of instances in this image. (instance-level masks), of shape (N, image_height, image_width)',
    'gt_boxes': 'bounding boxes and classes of instances in this image, of shape (N, 5), each entry is (x1, y1, x2, y2)',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)
  
  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
  print("**************\n",file_pattern,"***********")
  
  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader # just the class name
 
  keys_to_features = {
    'image/img_id': tf.FixedLenFeature((), tf.int64, default_value=0),
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/height': tf.FixedLenFeature((), tf.int64, default_value=224),
    'image/width': tf.FixedLenFeature((), tf.int64, default_value=224),
    'label/num_instances': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'label/gt_boxes': tf.FixedLenFeature((), tf.string),
    'label/gt_masks': tf.FixedLenFeature((), tf.string),
    'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
  }
  
  def _masks_decoder(keys_to_tensors):
    masks = tf.decode_raw(keys_to_tensors['label/gt_masks'], tf.uint8)
    width = tf.cast(keys_to_tensors['image/width'], tf.int32)
    height = tf.cast(keys_to_tensors['image/height'], tf.int32)
    instances = tf.cast(keys_to_tensors['label/num_instances'], tf.int32)
    mask_shape = tf.stack([instances, height, width])
    # print("Mask shape:",instances, height, width )
    return tf.reshape(masks, mask_shape)
  
  def _gt_boxes_decoder(keys_to_tensors):
    bboxes = tf.decode_raw(keys_to_tensors['label/gt_boxes'], tf.float32)
    instances = tf.cast(keys_to_tensors['label/num_instances'], tf.int32)
    bboxes_shape = tf.stack([instances, 5])
    return tf.reshape(bboxes, bboxes_shape)
  
  def _width_decoder(keys_to_tensors):
    width = keys_to_tensors['image/width']
    return tf.cast(width, tf.int32)
  
  def _height_decoder(keys_to_tensors):
    height = keys_to_tensors['image/height']
    return tf.cast(height, tf.int32)

  def _label_decoder(keys_to_tensors):
    label = tf.decode_raw(keys_to_tensors['label/encoded'], tf.uint8)
    width = tf.cast(keys_to_tensors['image/width'], tf.int32)
    height = tf.cast(keys_to_tensors['image/height'], tf.int32)
    img_shape = tf.stack([height, width, 1])
    return tf.reshape(label, img_shape)

  def _img_decoder(keys_to_tensors):
    image = tf.decode_raw(keys_to_tensors['image/encoded'], tf.uint8)
    width = tf.cast(keys_to_tensors['image/width'], tf.int32)
    height = tf.cast(keys_to_tensors['image/height'], tf.int32)
    img_shape = tf.stack([height, width, 3])
    # print("Mask shape:",instances, height, width )
    return tf.reshape(image, img_shape)
  
  items_to_handlers = {
    'image': slim.tfexample_decoder.ItemHandlerCallback(['image/encoded', 'image/height', 'image/width'],_img_decoder), 
    'label': slim.tfexample_decoder.ItemHandlerCallback(['label/encoded', 'image/height', 'image/width'],_label_decoder), 
    'gt_boxes': slim.tfexample_decoder.ItemHandlerCallback(['label/gt_boxes', 'label/num_instances'], _gt_boxes_decoder),
    'gt_masks': slim.tfexample_decoder.ItemHandlerCallback(['label/gt_masks', 'label/num_instances', 'image/width', 'image/height'], _masks_decoder),
  }
  
  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
  
  return slim.dataset.Dataset(
    data_sources=file_pattern,
    reader=reader,
    decoder=decoder,
    num_samples=SPLITS_TO_SIZES[split_name],
    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
    num_classes=_NUM_CLASSES)


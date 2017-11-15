#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function

from six.moves import xrange
import datetime

import functools
import math
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from deployment import model_deploy
from config.config import FLAGS
from nets.vgg_fcn import vgg_arg_scope
from nets.vgg_fcn import vgg_fcn

import datasets.dataset_factory as dataset_factory
import datasets.voc_preprocessing as _preprocessing
import utils.utils as utils

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

def main(_):
   '''
   training with optimization
   '''
   if not FLAGS.dataset_dir:
     raise ValueError('You must supply the dataset directory with --dataset_dir')

   tf.logging.set_verbosity(tf.logging.INFO)
   with tf.Graph().as_default():
       network_fn = get_network_fn(num_classes=FLAGS.num_classes, is_training=False)

       deploy_config = model_deploy.DeploymentConfig(
          num_clones=FLAGS.num_clones,
          clone_on_cpu=FLAGS.clone_on_cpu,
          replica_id=FLAGS.task,
          num_replicas=FLAGS.worker_replicas,
          num_ps_tasks=FLAGS.num_ps_tasks)
 
       with tf.device(deploy_config.variables_device()):
          global_step = slim.create_global_step()
       
       #train_set = dataset_factory.get_dataset(FLAGS.dataset_name, "train", FLAGS.dataset_dir)
       val_set = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

       val_provider = slim.dataset_data_provider.DatasetDataProvider(
          val_set,
          num_readers=FLAGS.num_readers,
          #reader_kwargs={'options':options},
          common_queue_capacity=20 * FLAGS.batch_size,
          common_queue_min=10 * FLAGS.batch_size)
       
       [val_image, val_label, val_boxes] = val_provider.get(['image', 'label', 'gt_boxes'])
       print("Validation image:",val_image, val_label, val_boxes) 
       val_image, val_label, val_boxes, val_masks = _preprocessing.preprocess_image(val_image, val_label, val_boxes)
       
       val_images, val_labels = tf.train.batch(
          [val_image, val_label],
          batch_size=FLAGS.batch_size,
          num_threads=FLAGS.num_preprocessing_threads,
          capacity=5 * FLAGS.batch_size)
       
###       val_batch_queue = slim.prefetch_queue.prefetch_queue(
       val_images = tf.squeeze(val_images,[1])
       annotation, fc8s, end_points = network_fn(images=val_images)


       summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
       summaries.add(tf.summary.image("Original_images", val_images))                    
       summaries.add(tf.summary.image("Ground_truth_masks", val_labels))
       summaries.add(tf.summary.image("Prediction_masks", tf.to_float(annotation)))

       for end_point in end_points: 
          x = end_points[end_point]
          summaries.add(tf.summary.histogram('activations/' + end_point, x))
          summaries.add(tf.summary.scalar('sparsity/' + end_point,
                        tf.nn.zero_fraction(x)))

       # Add summaries for variables.
       for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))

       names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
           'mean_iou': slim.metrics.streaming_mean_iou(annotation, val_labels, num_classes=FLAGS.num_classes),
           'precision': slim.metrics.streaming_accuracy(annotation, val_labels),
       })

       summary_op = tf.summary.merge(list(summaries), name='summary_op')


       for name, value in names_to_values.items():
          summary_name = 'eval/%s' % name
          op = tf.summary.scalar(summary_name, value, collections=[])
          op = tf.Print(op, [value], summary_name)
          tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

       if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
          variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
          variables_to_restore[tf_global_step.op.name] = tf_global_step
       else:
          variables_to_restore = slim.get_variables_to_restore()

       if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
       else:
          checkpoint_path = FLAGS.checkpoint_path

       tf.logging.info('Evaluating %s' % checkpoint_path)


       if FLAGS.max_num_batches:
          num_batches = FLAGS.max_num_batches
       else:
          # This ensures that we make a single pass over all of the data.
          num_batches = math.ceil(val_set.num_samples / float(FLAGS.batch_size))


       slim.evaluation.evaluate_once(
           master='',
           checkpoint_path=checkpoint_path,
           logdir=FLAGS.logs_dir,
           num_evals=num_batches,
           eval_op=list(names_to_updates.values()),
           variables_to_restore=variables_to_restore)
  
if __name__=="__main__":
   tf.app.run()

#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from six.moves import xrange
import datetime

import functools
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
       network_fn = get_network_fn(num_classes=FLAGS.num_classes, is_training=True)

       deploy_config = model_deploy.DeploymentConfig(
          num_clones=FLAGS.num_clones,
          clone_on_cpu=FLAGS.clone_on_cpu,
          replica_id=FLAGS.task,
          num_replicas=FLAGS.worker_replicas,
          num_ps_tasks=FLAGS.num_ps_tasks)
 
       with tf.device(deploy_config.variables_device()):
          global_step = slim.create_global_step()
       
       train_set = dataset_factory.get_dataset(FLAGS.dataset_name, "train", FLAGS.dataset_dir)
       #val_set = dataset_factory.get_dataset(FLAGS.dataset_name, "val", FLAGS.dataset_dir)


       with tf.device(deploy_config.inputs_device()):
          #####Consider Replace the following until #####
          #options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
          train_provider = slim.dataset_data_provider.DatasetDataProvider(
             train_set,
             num_readers=FLAGS.num_readers,
          #   reader_kwargs={'options':options},
             common_queue_capacity=20 * FLAGS.batch_size,
             common_queue_min=10 * FLAGS.batch_size)
          try:
             [train_image, train_label, train_boxes] = train_provider.get(['image', 'label', 'gt_boxes'])
          #[train_image, train_boxes, train_masks] = train_provider.get(['image', 'gt_boxes', 'gt_masks'])
             print(train_image, train_label, train_boxes)
             train_image, train_label, train_boxes, train_masks = _preprocessing.preprocess_image(train_image, train_label, train_boxes, is_training=True)
          except Exception as e:
             print(e)
             return
          train_images, train_labels = tf.train.batch(
             [train_image, train_label],
             batch_size=FLAGS.batch_size,
             num_threads=FLAGS.num_preprocessing_threads,
             capacity=5 * FLAGS.batch_size)
         
          train_batch_queue = slim.prefetch_queue.prefetch_queue(
             [train_images, train_labels], capacity=2 * FLAGS.num_clones)
          print(train_batch_queue)

          #val_provider = slim.dataset_data_provider.DatasetDataProvider(
          #   val_set,
          #   num_readers=FLAGS.num_readers,
          #   reader_kwargs={'options':options},
          #   common_queue_capacity=20 * FLAGS.batch_size,
          #   common_queue_min=10 * FLAGS.batch_size)
          #
          #[val_image, val_label, val_boxes, val_masks] = val_provider.get(['image', 'label', 'gt_boxes', 'gt_masks'])
          #  
          #val_image, val_label, val_boxes, val_masks = _preprocessing.preprocess_image(val_image, val_label, val_boxes, val_masks)
          #
          #val_images, val_labels = tf.train.batch(
          #   [val_image, val_label],
          #   batch_size=FLAGS.batch_size,
          #   num_threads=FLAGS.num_preprocessing_threads,
          #   capacity=5 * FLAGS.batch_size)
           
#          val_batch_queue = slim.prefetch_queue.prefetch_queue(
#             [val_images, val_labels], capacity=2 * FLAGS.num_clones)

       def clone_fn(batch_queue):
          """Allows data parallelism by creating multiple clones of networks"""
          images, labels = batch_queue.dequeue()
          #print(images, labels)
          images = tf.squeeze(images, [1])
          pred_annotation, fc8s, end_points = network_fn(images=images)
          ############################
          ## Loss function #
          ############################
          #print("Pred_annot", pred_annotation, "Labels", labels,"fc8s", fc8s)

          tf.losses.sparse_softmax_cross_entropy(logits=tf.to_float(pred_annotation),labels=tf.to_int32(labels), weights=1.0, scope="entropy")

          #loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(logits=tf.to_float(pred_annotation),labels=tf.to_int32(labels),scope="entropy")))

          return images, labels, pred_annotation, end_points

       summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
       clones = model_deploy.create_clones(deploy_config, clone_fn, [train_batch_queue])
       clone_scope = deploy_config.clone_scope(0)

       update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone_scope)

       images, labels, pred_annotation, end_points = clones[0].outputs

       summaries.add(tf.summary.image("Original_images", images))
       summaries.add(tf.summary.image("Ground_truth_masks", labels))
       summaries.add(tf.summary.image("Prediction_masks", tf.to_float(pred_annotation)))

       for end_point in end_points:                                             
          x = end_points[end_point]
          summaries.add(tf.summary.histogram('activations/' + end_point, x))
          summaries.add(tf.summary.scalar('sparsity/' + end_point,
                        tf.nn.zero_fraction(x)))

       for loss in tf.get_collection(tf.GraphKeys.LOSSES, clone_scope):
          summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

       # Add summaries for variables.
       for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))


       with tf.device(deploy_config.optimizer_device()):
          learning_rate = utils._configure_learning_rate(train_set.num_samples, global_step)
          optimizer = utils._configure_optimizer(learning_rate)
          summaries.add(tf.summary.scalar('learning rate', learning_rate))

       if FLAGS.sync_replicas:
          # If sync_replicas is enabled, the averaging will be done in the chief
          # queue runner.
          optimizer = tf.train.SyncReplicasOptimizer(
              opt=optimizer,
              replicas_to_aggregate=FLAGS.replicas_to_aggregate,
              variable_averages=variable_averages,
              variables_to_average=moving_average_variables,
              replica_id=tf.constant(FLAGS.task, tf.int32, shape=()),
              total_num_replicas=FLAGS.worker_replicas)
       elif FLAGS.moving_average_decay:
          # Update ops executed locally by trainer.
          update_ops.append(variable_averages.apply(moving_average_variables))

       variables_to_train = utils._get_variables_to_train()
       for var in variables_to_train:
          print(var.op.name)

       total_loss, clones_gradients = model_deploy.optimize_clones(
          clones,
          optimizer,
          var_list=variables_to_train)
       print('total_loss', total_loss, 'clone_gradients', clones_gradients)
       summaries.add(tf.summary.scalar('total_loss', total_loss))

       grad_updates = optimizer.apply_gradients(clones_gradients,global_step=global_step)

       update_ops.append(grad_updates)

       update_op = tf.group(*update_ops)

       train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

       summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, clone_scope))

       summary_op = tf.summary.merge(list(summaries), name='summary_op')

# Validate Set Evaluation options
       #network_fn_eval = get_network_fn(num_classes=NUM_OF_CLASSES, is_training=False)
       #print("val_images", val_images)
       #val_preds, fc8s, _ = network_fn(images=val_images)
       #names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
       #    'mean_iou': slim.metrics.streaming_mean_iou(val_preds, val_labels, num_classes=NUM_OF_CLASSES),
       #})

       #for name, value in names_to_values.items():
       #   summary_name = 'eval/%s' % name
       #   op = tf.summary.scalar(summary_name, value, collections=[])
       #   op = tf.Print(op, [value], summary_name)
       #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

       #if FLAGS.moving_average_decay:
       #   variable_averages = tf.train.ExponentialMovingAverage(
       #       FLAGS.moving_average_decay, tf_global_step)
       #   variables_to_restore = variable_averages.variables_to_restore(
       #       slim.get_model_variables())
       #   variables_to_restore[tf_global_step.op.name] = tf_global_step
       #else:
       #   variables_to_restore = slim.get_variables_to_restore()

       #for i in range(FLAGS.max_steps / FLAGS.iter_train_steps):
       
       slim.learning.train(
           train_tensor,
           logdir=FLAGS.logs_dir,
           master='',
           is_chief=(FLAGS.task == 0),
           init_fn=utils._get_init_fn(),
           summary_op=summary_op,
           number_of_steps=FLAGS.max_steps, #FLAGS.iter_train_steps*(i+1) if FLAGS.max_steps > FLAGS.iter_train_steps*(i+1) else FLAGS.max_steps,
           log_every_n_steps=FLAGS.log_every_n_steps,
           save_summaries_secs=FLAGS.save_summaries_secs,
           save_interval_secs=FLAGS.save_interval_secs,
           sync_optimizer=optimizer if FLAGS.sync_replicas else None)

       #if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
       #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
       #else:
       #   checkpoint_path = FLAGS.checkpoint_path

       #tf.logging.info('Evaluating %s' % checkpoint_path)
       #slim.evaluation.evaluate_once(
       #    master='',
       #    checkpoint_path=checkpoint_path,
       #    logdir=FLAGS.logs_dir,
       #    num_evals=num_batches,
       #    eval_op=list(names_to_updates.values()),
       #    variables_to_restore=variables_to_restore)
   

if __name__=="__main__":
   tf.app.run()

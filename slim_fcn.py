#!/usr/bin/env python
#-*- coding:utf-8 -*-

from __future__ import print_function
import tensorflow as tf
import numpy as np
from six.moves import xrange
#import data_file_reader as datareader
from tensorflow.python.ops import control_flow_ops
import datetime
from deployment import model_deploy
import datasets.dataset_factory as dataset_factory
import datasets.coco_preprocessing as coco_preprocessing
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType

slim = tf.contrib.slim


tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
   'task', 0, 'task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'coco', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train2014', 'The split set of dataset')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')




## added by Yang ##
tf.app.flags.DEFINE_string(
    'logs_dir', 'tf_dir',
    'Logs directory for training and evaluation.')
tf.app.flags.DEFINE_integer(
    'num_clones', 1,
    'number of GPU duplicate clones')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'number of readers for dataset provider.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'number of preprocessing threads.')
tf.app.flags.DEFINE_float(
    'mean', 120.00,
    'mean of images, need to be substract from given images.')


FLAGS = tf.app.flags.FLAGS

MAX_ITERATION = int(1e3 + 1)
NUM_OF_CLASSES = 81
IMAGE_SIZE = 224

def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

def _get_init_fn():
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if FLAGS.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the logs_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(FLAGS.logs_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % FLAGS.logs_dir)
    return None

  exclusions = []
  if FLAGS.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
  else:
    checkpoint_path = FLAGS.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=FLAGS.ignore_missing_vars)

def _get_variables_to_train():
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if FLAGS.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train

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
     
def vgg_fcn(inputs,
            num_classes=NUM_OF_CLASSES,
            is_training=True,
            dropout_keep_prob=0.5,
            spatial_squeeze=False, #True,
            scope='vgg_16',
            fc_conv_padding='VALID'):
 
   """
    define slim form vgg
   """
   with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
      end_points_collection = sc.name + '_end_points'
      with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
         conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
         pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
         conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
         pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
         conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
         pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
         conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
         pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')
         conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], scope='conv5')
         pool5 = slim.max_pool2d(conv5, [2, 2], scope='pool5')
         fc6 = slim.conv2d(pool5, 4096, [7, 7],padding=fc_conv_padding, scope='fc6')
         drop6 = slim.dropout(fc6, dropout_keep_prob, is_training=is_training, scope='dropout6')
         fc7 = slim.conv2d(drop6, 4096, [1, 1], scope='fc7')
         drop7 = slim.dropout(fc7, dropout_keep_prob, is_training=is_training, scope='dropout7')
         fc8 = slim.conv2d(drop7, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
         #end_points = slim.utils.convert_collection_to_dict(end_points_collection)

         if spatial_squeeze:
            fc8 = tf.squeeze(fc8, [1, 2], name='fc8/squeezed')
         deconv_shape1 = pool4.get_shape() 
         print("Pool4:", deconv_shape1)
         # it is intialized as bilinar interpolation(calculation predfined)
         conv_t1 = slim.conv2d_transpose(inputs=fc7, num_outputs=deconv_shape1[3].value, kernel_size=[3, 3], stride=14, scope="conv_t1")
         fc32s = tf.add(conv_t1, pool4, name="fc32s")
         
         deconv_shape2 = pool3.get_shape()
         print(fc32s, deconv_shape2)
         conv_t2 = slim.conv2d_transpose(inputs=fc32s, num_outputs=deconv_shape2[3].value, kernel_size=[2, 2], stride=2, scope="conv_t2")
         fc16s = tf.add(conv_t2, pool3, name="fc16s")

         fc8s = conv_t3 = slim.conv2d_transpose(inputs=fc16s, num_outputs=NUM_OF_CLASSES, kernel_size=[3,3], stride=8, scope="fc8s")
         annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")
         end_points = slim.utils.convert_collection_to_dict(end_points_collection)
         print("vgg_net layers:", end_points.keys())

         tf.get_variable_scope().reuse_variables()

         return tf.expand_dims(annotation_pred, dim=3), fc8s, end_points


##  def train(loss_func, variables_list, optimizer):
##     '''
##       training operation definition, including optimizer, gradient_calculation
##     '''
##  
##     grads = optimizer.compute_gradients(loss_func, var_list = variables_list)
##  
##     return optimizer.apply_gradients(grads) if optimizer else None
##    
def main(_):
   '''
   training with optimization
   '''
   if not FLAGS.dataset_dir:
     raise ValueError('You must supply the dataset directory with --dataset_dir')

   tf.logging.set_verbosity(tf.logging.INFO)
   with tf.Graph().as_default():
       deploy_config = model_deploy.DeploymentConfig(
          num_clones=FLAGS.num_clones,
          clone_on_cpu=FLAGS.clone_on_cpu,
          replica_id=FLAGS.task,
          num_replicas=FLAGS.worker_replicas,
          num_ps_tasks=FLAGS.num_ps_tasks)
 
       with tf.device(deploy_config.variables_device()):
          global_step = slim.create_global_step()
       
       #train_batch_queue = None
       #val_batch_queue = None
       train_set = dataset_factory.get_dataset(FLAGS.dataset_name, "train2014", FLAGS.dataset_dir)
       #val_set = dataset_factory.get_dataset(FLAGS.dataset_name, "train2014", FLAGS.dataset_dir)


       with tf.device(deploy_config.inputs_device()):
          #####Consider Replace the following until #####
          options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
          train_provider = slim.dataset_data_provider.DatasetDataProvider(
             train_set,
             num_readers=FLAGS.num_readers,
             reader_kwargs={'options':options},
             common_queue_capacity=20 * FLAGS.batch_size,
             common_queue_min=10 * FLAGS.batch_size)
          [train_image, train_label, train_boxes, train_masks] = train_provider.get(['image', 'label', 'gt_boxes', 'gt_masks'])
          ##train_image=tf.reshape(train_image,(height, width, 3))
          ##train_label=tf.reshape(train_label,(height, width, 1))
         
          print(train_image, train_label, train_masks, train_boxes) 
          train_image, train_label, train_boxes, train_masks = coco_preprocessing.preprocess_image(train_image, train_label, train_boxes, train_masks, is_training=True)
          
          train_images, train_labels = tf.train.batch(
             [train_image, train_label],
             batch_size=FLAGS.batch_size,
             num_threads=FLAGS.num_preprocessing_threads,
             capacity=5 * FLAGS.batch_size)
         
          train_batch_queue = slim.prefetch_queue.prefetch_queue(
             [train_images, train_labels], capacity=2 * FLAGS.num_clones)
          print(train_batch_queue)

#          val_provider = slim.dataset_data_provider.DatasetDataProvider(
#             val_set,
#             num_readers=FLAGS.num_readers,
#             common_queue_capacity=20 * FLAGS.batch_size,
#             common_queue_min=10 * FLAGS.batch_size)
#          
#          [val_image, val_label, val_boxes, val_masks] = val_provider.get(['image', 'label', 'gt_boxes', 'gt_masks'])
#            
#          val_image, val_label, val_boxes, val_masks = coco_preprocessing.preprocess_image(val_image, val_label, val_boxes, val_masks)
#          
#          val_images, val_labels = tf.train.batch(
#             [val_image, val_label],
#             batch_size=FLAGS.batch_size,
#             num_threads=FLAGS.num_preprocessing_threads,
#             capacity=5 * FLAGS.batch_size)
#           
#          val_batch_queue = slim.prefetch_queue.prefetch_queue(
#             [val_images, val_labels], capacity=2 * FLAGS.num_clones)
#          print(val_batch_queue)

       def clone_fn(batch_queue):
          """Allows data parallelism by creating multiple clones of networks"""
          images, labels = batch_queue.dequeue()
          print(images, labels)
          images = tf.squeeze(images, [1])
          pred_annotation, fc8s, end_points = vgg_fcn(inputs=images)
          ############################
          ## Loss function #
          ############################
          #slim.losses.softmax_cross_entropy(pred_annotation, labels, label_smoothing=True, weights=1.0)
          print("Pred_annot", pred_annotation, "Labels", labels,"fc8s", fc8s)

          tf.losses.sparse_softmax_cross_entropy(logits=tf.to_float(fc8s),labels=tf.to_int32(labels), weights=1.0, scope="entropy")



          #loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(logits=tf.to_float(pred_annotation),labels=tf.to_int32(labels),scope="entropy")))

          return end_points

       summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
       clones = model_deploy.create_clones(deploy_config, clone_fn, [train_batch_queue])
       #clones2 = model_deploy.create_clones(deploy_config, clone_fn, [val_batch_queue])
       first_clone_scope = deploy_config.clone_scope(0)
       #second_clone_scope = deploy_config.clone_scope(0)

       update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

       end_points = clones[0].outputs
       for end_point in end_points:                                             
          x = end_points[end_point]
          summaries.add(tf.summary.histogram('activations/' + end_point, x))
          summaries.add(tf.summary.scalar('sparsity/' + end_point,
                        tf.nn.zero_fraction(x)))

       for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
          summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

       # Add summaries for variables.
       for variable in slim.get_model_variables():
          summaries.add(tf.summary.histogram(variable.op.name, variable))


       with tf.device(deploy_config.optimizer_device()):
          learning_rate = _configure_learning_rate(train_set.num_samples, global_step)
          optimizer = _configure_optimizer(learning_rate)
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

       variables_to_train = _get_variables_to_train()
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

       summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

#       summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, second_clone_scope))

       summary_op = tf.summary.merge(list(summaries), name='summary_op')

       slim.learning.train(
           train_tensor,
           logdir=FLAGS.logs_dir,
           master='',
           is_chief=(FLAGS.task == 0),
           init_fn=_get_init_fn(),
           summary_op=summary_op,
           number_of_steps=FLAGS.max_steps,
           log_every_n_steps=FLAGS.log_every_n_steps,
           save_summaries_secs=FLAGS.save_summaries_secs,
           save_interval_secs=FLAGS.save_interval_secs,
           sync_optimizer=optimizer if FLAGS.sync_replicas else None)


       ##image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
       ##keep_probability = tf.placeholder(tf.float32, name="keep_probability")
       ##annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
       ##
       ##pred_annotation, logits, end_points = vgg_fcn(inputs=image, dropout_keep_prob=keep_probability)
       ##tf.summary.image("input_image", image, max_outputs=4)
       ##tf.summary.image("groud_truth", tf.cast(annotation, tf.uint8), max_outputs=4)
       ##tf.summary.image("prediction", tf.cast(pred_annotation, tf.uint8), max_outputs=4)
       ##
       ##losses = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=tf.to_int32(annotation), scope='entropy')))
       ##with tf.devices(deploy_config.variable_device()):
       ##    trainable_variables = _get_variables_to_train()
       ##    learning_rate = _configure_learning_rate(train_set.num_samples, global_step)
       ##    optimizer = _configure_optimizer(learning_rate)
       ##tf.summary.scalar('learning_rate', learning_rate)

       ##train_op = train(losses, trainable_variables, optimizer)
       ##
       ##summary_op = tf.summary.merge_all()
       ##
       ##with tf.Session() as sess:
       ##    print("Setting up Saver...")
       ##    saver = tf.train.Saver()
       ##    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)    
       ##    sess.run(tf.global_variables_initializer())
       ##    print("Initialize all params...")
       ##    if FLAGS.checkpoint_exclude_scopes:
       ##        exclusions = [scope.strip()
       ##                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
       ##    variables_to_restore = []
       ##    for var in slim.get_model_variables():
       ##       excluded = False
       ##       for exclusion in exclusions:
       ##          if var.op.name.startswith(exclusion):
       ##             excluded = True
       ##             break
       ##       if not excluded:
       ##          variables_to_restore.append(var)

       ##    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
       ##       checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
       ##    else:
       ##       checkpoint_path = FLAGS.checkpoint_path

       ##    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)                 
       ##    if ckpt and ckpt.model_checkpoint_path:
       ##       saver.restore(sess, ckpt.model_checkpoint_path)
       ##       print("Ignoring original checkpoint because a newer one exists!")
       ##    elif checkpoint_path:
       ##       # saver.restore(sess, checkpoint_path)
       ##       slim.assign_from_checkpoint_fn(
       ##            checkpoint_path,
       ##            variables_to_restore,
       ##            ignore_missing_vars=True)
       ##       print("Model restored...")

       ##    saver.save(sess, FLAGS.checkpoint_path + "model.ckpt")
       ##    for itr in xrange(MAX_ITERATION):
       ##       print("%dth iteration" % itr)
       ##       train_images, train_annotations = train_batch_queue.dequeue()
       ##       print("Batch get!")
       ##       print(tf.shape(train_images), tf.shape(train_annotations))
       ##       train_images, train_annotations = sess.run([train_images, train_annotations])
       ##       print("Images to numpy!")

       ##       feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.8}
       ##       print("Train running")
       ##       sess.run([train_op, pred_annotation, losses], feed_dict=feed_dict)
       ##       print("Success")
       ##       if itr % 50 == 0:
       ##          train_loss, summary_str = sess.run([losses, summary_op], feed_dict=feed_dict)
       ##          print("Step: %d, Train_loss:%g" % (itr, train_loss))
       ##          summary_writer.add_summary(summary_str, itr)
       ## 
       ##       if itr % 500 == 0:
       ##          valid_images, valid_annotations = clones2 #val_batch_queue.dequeue()
       ##          valid_loss = sess.run([losses], feed_dict={image:valid_images, annotation:valid_annotations, keep_probability:1.0})
       ##          print("%s ---> Validation_loss: %g" %(datetime.datetime.now(), valid_loss))
       ##          saver.save(sess, FLAGS.checkpoint_path + "model.ckpt", itr)

if __name__=="__main__":
   tf.app.run()

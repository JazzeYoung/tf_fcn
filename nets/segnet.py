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

def segnet_arg_scope():
    


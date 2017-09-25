from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from datasets import coco
#from libs.datasets import city
from datasets import coco_preprocessing as coco_preprocess

def get_dataset(dataset_name, split_name, dataset_dir, 
        im_batch=1, is_training=False, file_pattern=None, reader=None):

    return coco.get_split(
             split_name,
             dataset_dir,
             file_pattern,
             reader)

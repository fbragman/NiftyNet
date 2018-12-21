# -*- coding: utf-8 -*-
"""
This class allows one to convert an image of floats into a binned uint8 image. This
allows one to convert a regression task into a segmentation task where the aim is
to predict the bin (0-255) of a voxel, rather than regress a float. Given the training set,
we find the average CDF mapping for equalisation. Each image is equalised given this mapping,
then binned into (0-255).
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_equalisation as he
from niftynet.layer.base_layer import DataDependentLayer
from niftynet.utilities.util_common import print_progress_bar


class DataBinningLayer(DataDependentLayer):

    def __init__(self,
                 image_name,
                 model_dir,
                 name='float2int'):
        """

        :param image_name:
        :param name:
        """
        super(DataBinningLayer, self).__init__(name=name)
        self.image_name = image_name
        self.model_dir = model_dir

        self.train_min = None
        self.train_max = None

    def layer_op(self, image):

        if isinstance(image, dict):
            image_5d = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_5d = np.asarray(image, dtype=np.float32)

        # equalisation
        # conversion to uint8

    def train(self, image_list):

    def is_ready(self):

    def __check_modalities_to_train(self):




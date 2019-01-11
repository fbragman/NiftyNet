# -*- coding: utf-8 -*-
"""
This class allows one to convert an image of floats into a binned uint8 image. This
allows one to convert a regression task into a segmentation task where the aim is
to predict the bin (0-255) of a voxel, rather than regress a float. Given the training set,
we find the average CDF mapping for equalisation. Each image is equalised given this mapping,
then binned into (0-255). A regularised equalisation is performed by adding a small value to each
bin of the population PMF before conversion. The inverse CDF can be used to de-equalise any image
in order to convert an integer volume into an original raw HU volume.
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf

import niftynet.utilities.histogram_equalisation as he
from niftynet.layer.base_layer import DataDependentLayer
from niftynet.layer.base_layer import Invertible
from niftynet.layer.binary_masking import BinaryMaskingLayer


class HistogramEqualisationBinningLayer(DataDependentLayer, Invertible):
    def __init__(self,
                 image_name,
                 model_filename=None,
                 name='hist_equal'):
        """
        :param image_name:
        :param model_filename
        :param name:
        """
        super(HistogramEqualisationBinningLayer, self).__init__(name=name)

        if model_filename is None:
            model_filename = os.path.join('.', 'cdf_mapping_file.txt')
        self.model_file = os.path.abspath(model_filename)
        assert not os.path.isdir(self.model_file), \
            "model_filename is a directory, " \
            "please change histogram_ref_file to a filename."

        self.image_name = image_name
        self.mapping = he.read_mapping_file(self.model_file)

    def layer_op(self, image, mask=None):
        if isinstance(image, dict):
            image_5d = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_5d = np.asarray(image, dtype=np.float32)

        if isinstance(mask, dict):
            image_mask = mask.get(self.image_name, None)
        elif mask is not None:
            image_mask = mask
        elif self.binary_masking_func is not None:
            image_mask = self.binary_masking_func(image_5d)
        else:
            # no access to mask, default to all image
            image_mask = np.ones_like(image_5d, dtype=np.bool)

        result = self._equalise_to_int(image_5d, image_mask)

        if isinstance(image, dict):
            image[self.image_name] = result
            return image, mask
        else:
            return result, mask

    def inverse_op(self, image):
        assert self.is_ready(), \
            "histogram equalisation layer needs to be trained first."
        if isinstance(image, dict):
            image_5d = np.asarray(image[self.image_name], dtype=np.float32)
        else:
            image_5d = np.asarray(image, dtype=np.float32)

        result = self._int_to_deequalise(image_5d)
        if isinstance(image, dict):
            image[self.image_name] = result
            return image
        else:
            return result

    def train(self, image_list):
        # check modalities to train, using the first subject in subject list
        # to find input modality list
        #if self.is_ready():
        #    tf.logging.info(
        #        "normalisation equalisation model ready")
        #    return
        mapping = he.create_cdf_mapping_from_arrayfiles(image_list, self.image_name)
        int_mapping = he.create_int_mapping_from_arrayfiles(image_list[0][self.image_name], mapping)
        mapping.update(int_mapping)

        he.write_mapping(self.model_file, mapping)

    def is_ready(self):
        raise NotImplemented

    def _int_to_deequalise(self, img_data):

        return he.transform_by_inverse_mapping(img_data, self.mapping)

    def _equalise_to_int(self, img_data, mask):
        assert img_data.ndim == 5

        if not self.mapping:
            tf.logging.fatal(
                "calling equaliser with empty mapping,"
                "probably {} is not loaded".format(self.model_file))
            raise RuntimeError
        mask_array = np.asarray(mask, dtype=np.bool)


        return he.transform_by_mapping(img_data, self.mapping)


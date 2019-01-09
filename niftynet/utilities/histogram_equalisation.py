# -*- coding: utf-8 -*-
"""
Average equalisation mapping
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import os
import tensorflow as tf
from skimage import exposure

def read_mapping_file(mapping_file):
    """
    Reads an existing mapping file with the given modalities.
    :param mapping_file: file in which mapping is stored
    :return mapping_dict: dictionary containing the mapping landmarks for
    each modality stated in the mapping file
    """
    mapping_dict = {}
    if not mapping_file:
        return mapping_dict
    if not os.path.isfile(mapping_file):
        return mapping_dict

    with open(mapping_file, "r") as f:
        for line in f:
            if len(line) <= 2:
                continue
            line = line.split()
            if len(line) < 2:
                continue
            try:
                map_name, map_value = line[0], np.float32(line[1:])
                mapping_dict[map_name] = tuple(map_value)
            except ValueError:
                tf.logging.fatal(
                    "unknown input format: {}".format(mapping_file))
                raise
    return mapping_dict

def _regularised_cdf(image, nbins, d=0.0001):
    hist, bin_centers = exposure.histogram(image, nbins)
    hist = hist + d * np.sum(hist)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])

    return img_cdf, bin_centers

def _inverse_digitize_map(int_image, float_to_int_bin_mapping):
    new_img = np.float32(int_image)

    bin_diff = np.diff(float_to_int_bin_mapping)[0] / 2
    bin_vals = float_to_int_bin_mapping + bin_diff
    bin_vals[-1] = 1.0

    for bin_iter, bin_val in enumerate(bin_vals):
        new_img[int_image == bin_iter] = bin_val

    return new_img

def transform_by_mapping(img, mapping_file):

def transform_by_inverse_mapping(img, mapping_file):

def create_mapping_from_arrayfiles(arrayfiles):
    """
    Performs the mapping creation based on a list of files. The cdf for the entire set
    of images is calculated and used to equalise images. The equalised images are then
    quantised into the range [0, 255].
    :param arrayfiles:
    :return:
    """

def write_mapping(mapping_file, mapping):
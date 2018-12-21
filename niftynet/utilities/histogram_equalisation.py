# -*- coding: utf-8 -*-
"""
Average equalisation mapping
"""
from __future__ import absolute_import, print_function, division
from skimage.exposure import cumulative_distribution

import numpy as np


def create_mapping_from_arrayfiles(array_files, nbins=1000):
    """
    Performs the mapping creation based on a list of files.
    :param array_files:
    :return:
    """
    images = []
    for array_file in array_files:
        images.append(array_file)

    stacked_images = np.concatenate(images, axis=-1)
    training_cdf, bin_centers = cumulative_distribution(stacked_images, nbins=nbins)

    return training_cdf, bin_centers


def transform_by_mapping(img, training_cdf, bin_centers):
    """
    Perform image equalisation given cdf and bins
    :param img:
    :param training_cdf:
    :param bin_centers:
    :return:
    """
    img_eq = np.interp(img, bin_centers, training_cdf)
    return img_eq


def transform_by_inverse_mapping(img_eq, training_cdf, bin_centers):
    """
    Perform image de-equalisation given training cdf and bins
    :param img:
    :param training_cdf:
    :param bin_centers:
    :return:
    """
    img = np.interp(img_eq, training_cdf, bin_centers)
    return img

def read_mapping_file(mapping_file):
    """
    Read mapping file given training set
    :return:
    """
    return None

# -*- coding: utf-8 -*-
"""
Average equalisation and integer mapping
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import os
import tensorflow as tf
from skimage import exposure
from niftynet.utilities.util_common import print_progress_bar
from niftynet.io.misc_io import touch_folder

float2intbins = 256


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


def _regularised_cdf(images, nbins=2000, d=0.0001):
    hist, bin_centers = exposure.histogram(images, nbins)
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

    img_eqd = np.interp(img, mapping_file['bin_centers'],
                             mapping_file['fwd_cdf'])
    img_bin = np.digitize(img_eqd, mapping_file['int_mapping']).astype(np.uint8)

    return img_bin


def transform_by_inverse_mapping(img, mapping_file):

    img_convd_float = _inverse_digitize_map(img, mapping_file['int_mapping'])
    de_eqd_img = np.interp(img_convd_float, mapping_file['fwd_cdf'],
                                            mapping_file['bin_centers'])

    return de_eqd_img


def create_int_mapping_from_arrayfiles(array_file, mapping):
    """
    Calculates bin edges of equalised float to uint8 conversion. Only
    necessary to use an image from array_files as range of intensities
    after using the population CDF will be equivalent across all images
    :param array_files:
    :param mapping
    :return:
    """
    tmp_eqd_img = np.interp(array_file.get_data(), mapping['bin_centers'],
                                                   mapping['fwd_cdf'])
    float_to_int_bin_mapping = np.linspace(tmp_eqd_img.min(),
                                           tmp_eqd_img.max(),
                                           float2intbins)

    mapping = dict()
    mapping['int_mapping'] = float_to_int_bin_mapping

    return mapping


def create_cdf_mapping_from_arrayfiles(array_files, field):
    """
    Performs the mapping creation based on a list of files. The cdf for the entire set
    of images is calculated and used to equalise images.
    :param array_files:
    :return:
    """

    stacked_list = []
    for i, p in enumerate(array_files):
        print_progress_bar(i, len(array_files),
                           prefix='histogram equalisation training',
                           decimals=1, length=10, fill='*')
        tmp_data = p[field].get_data()
        stacked_list.append(tmp_data)

    # Concatenate all images
    stacked_img = np.concatenate(stacked_list, axis=-1)

    # Get regularised CDF: mapping function
    cdf_map, bin_map = _regularised_cdf(stacked_img)
    mapping = dict()
    mapping['fwd_cdf'] = cdf_map
    mapping['bin_centers'] = bin_map

    return mapping


def write_mapping(mapping_file, mapping):

    # backup existing file first
    if os.path.exists(mapping_file):
        backup_name = '{}.backup'.format(mapping_file)
        from shutil import copyfile
        try:
            copyfile(mapping_file, backup_name)
        except OSError:
            tf.logging.warning('cannot backup file {}'.format(mapping_file))
            raise
        tf.logging.warning(
            "moved existing histogram reference file\n"
            " from {} to {}".format(mapping_file, backup_name))

    touch_folder(os.path.dirname(mapping_file))
    __force_writing_new_mapping(mapping_file, mapping)


def __force_writing_new_mapping(filename, mapping_dict):
    """
    Writes a mapping dictionary to file

    :param filename: name of the file in which to write the saved mapping
    :param mapping_dict: mapping dictionary to save in the file
    :return:
    """
    with open(filename, 'w+') as f:
        for mod in mapping_dict.keys():
            mapping_string = ' '.join(map(str, mapping_dict[mod]))
            string_fin = '{} {}\n'.format(mod, mapping_string)
            f.write(string_fin)
    return

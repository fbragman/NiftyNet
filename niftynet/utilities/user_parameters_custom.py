# -*- coding: utf-8 -*-
"""
This module defines task specific parameters
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from niftynet.utilities.user_parameters_helper import add_input_name_args
from niftynet.utilities.user_parameters_helper import int_array
from niftynet.utilities.user_parameters_helper import str2boolean


#######################################################################
# To support a CUSTOM_SECTION in config file:
# (e.g., MYTASK; in parallel with SEGMENTATION, REGRESSION etc.)
#
# 1) update niftynet.utilities.user_parameters_custom.SUPPORTED_ARG_SECTIONS
# with a key-value pair:
# where the key should be MYTASK, a standardised string --
# Standardised string is defined in
# niftynet.utilities.user_parameters_helper.standardise_string
# the section name will be filtered with,
# re.sub('[^0-9a-zA-Z_\- ]+', '', input_string.strip())
#
# the value should be __add_mytask_args()
#
# 2) create a function __add_mytask_args() with task specific arguments
# this function should return an argparse obj
#
# 3) in the application file, specify:
# `REQUIRED_CONFIG_SECTION = "MYTASK"`
# so that the application will have access to the task specific arguments
#########################################################################


def add_customised_args(parser, task_name):
    """
    loading keywords arguments to parser by task name
    :param parser:
    :param task_name: supported choices are listed in `SUPPORTED_ARG_SECTIONS`
    :return: parser with updated actions
    """
    task_name = task_name.upper()
    if task_name in SUPPORTED_ARG_SECTIONS:
        return SUPPORTED_ARG_SECTIONS[task_name](parser)
    else:
        raise NotImplementedError


def __add_regression_args(parser):
    """
    keywords defined for regression tasks

    :param parser:
    :return:
    """
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    parser.add_argument(
        "--error_map",
        metavar='',
        help="Set whether to output the regression error maps (the maps "
             "will be stored in $model_dir/error_maps; the error maps "
             "can be used for window sampling).",
        type=str2boolean,
        default=False)

    from niftynet.application.regression_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_segmentation_args(parser):
    """
    keywords defined for segmentation tasks

    :param parser:
    :return:
    """
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--softmax",
        metavar='',
        help="[Training only] whether to append a softmax layer to network "
             "output before feeding it into loss function",
        type=str2boolean,
        default=True)

    # for selective sampling only
    parser.add_argument(
        "--min_sampling_ratio",
        help="[Training only] Minimum ratio of samples in a window for "
             "selective sampler",
        metavar='',
        type=float,
        default=0
    )

    # for selective sampling only
    parser.add_argument(
        "--compulsory_labels",
        help="[Training only] List of labels to have in the window for "
             "selective sampling",
        metavar='',
        type=int_array,
        default=(0, 1)
    )

    # for selective sampling only
    parser.add_argument(
        "--rand_samples",
        help="[Training only] Number of completely random samples per image "
             "when using selective sampler",
        metavar='',
        type=int,
        default=0
    )

    # for selective sampling only
    parser.add_argument(
        "--min_numb_labels",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=int,
        default=1
    )

    # for selective sampling only
    parser.add_argument(
        "--proba_connect",
        help="[Training only] Number of labels to have in the window for "
             "selective sampler",
        metavar='',
        type=str2boolean,
        default=True
    )

    parser.add_argument(
        "--evaluation_units",
        help="Compute per-component metrics for per label or per connected "
             "component. [foreground, label, or cc]",
        choices=['foreground', 'label', 'cc'],
        default='foreground')

    from niftynet.application.segmentation_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_gan_args(parser):
    """
    keywords defined for GAN

    :param parser:
    :return:
    """
    parser.add_argument(
        "--noise_size",
        metavar='',
        help="length of the noise vector",
        type=int,
        default=-1)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.gan_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_classification_args(parser):
    """
    keywords defined for classification

    :param parser:
    :return:
    """
    parser.add_argument(
        "--num_classes",
        metavar='',
        help="Set number of classes",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_prob",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)


    from niftynet.application.classification_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_autoencoder_args(parser):
    """
    keywords defined for autoencoder

    :param parser:
    :return:
    """
    from niftynet.application.autoencoder_application import SUPPORTED_INFERENCE
    parser.add_argument(
        "--inference_type",
        metavar='',
        help="choose an inference type_str for the trained autoencoder",
        choices=list(SUPPORTED_INFERENCE))

    parser.add_argument(
        "--noise_stddev",
        metavar='',
        help="standard deviation of noise when inference type_str is sample",
        type=float)

    parser.add_argument(
        "--n_interpolations",
        metavar='',
        help="the method of generating window from image",
        type=int,
        default=10)

    from niftynet.application.autoencoder_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_registration_args(parser):
    """
    keywords defined for image registration

    :param parser:
    :return:
    """
    parser.add_argument(
        "--label_normalisation",
        metavar='',
        help="whether to map unique labels in the training set to "
             "consecutive integers (the smallest label will be  mapped to 0)",
        type=str2boolean,
        default=False)

    from niftynet.application.label_driven_registration import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


def __add_multitask_args(parser):
    parser.add_argument(
        "--loss_border",
        metavar='',
        help="Set the border size for the loss function to ignore",
        type=int,
        default=0)

    parser.add_argument(
        "--num_classes",
        help="Set number of classes for tasks as nc_1, nc_2",
        type=int_array,
        default=-1)

    parser.add_argument(
        "--loss_task_1",
        metavar='',
        type=str,
        help="[Training only] loss function for task 1 type_str",
        default='RMSE')

    parser.add_argument(
        "--loss_task_2",
        metavar='',
        type=str,
        help="[Training only] loss function for task 2 type_str",
        default='CrossEntropy')

    parser.add_argument(
        "--task_1_type",
        metavar='',
        type=str,
        help="[Training only] loss function for task 1 type_str",
        default='regression')

    parser.add_argument(
        "--task_2_type",
        metavar='',
        type=str,
        help="[Training only] loss function for task 2 type_str",
        default='classification')

    # Categorical sampling parameters
    parser.add_argument(
        "--use_hardcat",
        metavar='',
        type=str2boolean,
        help="[Method option] hard parameter in GumbelSoftmax class, "
             "if hard=True, hard stochastic in fwd pass with GS approx in bwd pass",
        default=True)

    parser.add_argument(
        "--categorical",
        metavar='',
        type=str2boolean,
        help="[Method option] if True, sample from a categorical over "
             "the learned parameters p. If False, use the learned p as soft weights",
        default=True)

    # Merging of tensors per layer
    parser.add_argument(
        "--group_connection",
        metavar='',
        type=str,
        help="[Method option] sets structural options for multi-task",
    )

    # Gumbel-Softmax annealing parameters
    parser.add_argument(
        "--use_tau_annealing",
        metavar='',
        type=str2boolean,
        help="[Method option] if True, annealing of temperature used for"
             "Gumbel-Softmax approximation",
        default=False)

    parser.add_argument(
        "--tau",
        help="[Method option] Initial temperature or constant temperature for"
             "Gumbel-Softmax approximation",
        type=float,
        default=1
    )

    parser.add_argument(
        "--gs_anneal_r",
        help="[Method option] GumbelSoftmax annealing hyper-parameter r",
        type=float,
        default=0.0001
    )

    parser.add_argument(
        "--output_interp_order_task1",
        help='Output for task 1',
        type=int,
        default=3)

    parser.add_argument(
        "--output_interp_order_task2",
        help='Output for task 2',
        type=int,
        default=0)

    parser.add_argument(
        "--output_prob_task_1",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities for task 1",
        type=str2boolean,
        default=False)

    parser.add_argument(
        "--output_prob_task_2",
        metavar='',
        help="[Inference only] whether to output multi-class probabilities for task 2",
        type=str2boolean,
        default=False)

    from niftynet.application.multitask_application import SUPPORTED_INPUT
    parser = add_input_name_args(parser, SUPPORTED_INPUT)
    return parser


SUPPORTED_ARG_SECTIONS = {
    'REGRESSION': __add_regression_args,
    'SEGMENTATION': __add_segmentation_args,
    'CLASSIFICATION': __add_classification_args,
    'AUTOENCODER': __add_autoencoder_args,
    'GAN': __add_gan_args,
    'REGISTRATION': __add_registration_args,
    'MULTITASK': __add_multitask_args
}

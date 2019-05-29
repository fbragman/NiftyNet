# -*- coding: utf-8 -*-
"""
Loss functions for bayesian segmentation loss functions
Support for:    i) Stochastic Cross-Entropy
                ii) Scaled Cross-Entropy
                iii) Scaled Cross-Entropy with numerical approximation
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf
import tensorflow_probability as tfp

from niftynet.engine.application_factory import LossSegmentationFactory
from niftynet.layer.base_layer import Layer


class LossBayesianSegFunction(Layer):
    def __init__(self,
                 n_class,
                 loss_type='ScaledCrossEntropyApprox',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossBayesianSegFunction, self).__init__(name=name)
        assert n_class > 0, \
            "Number of classes for segmentation loss should be positive."
        self._num_classes = n_class

        # set loss function and function-specific additional params.
        self._data_loss_func = LossSegmentationFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else dict()

        data_loss_function_name = self._data_loss_func.__name__
        if data_loss_function_name.startswith('cross_entropy') \
                or 'xent' in data_loss_function_name:
            tf.logging.info(
                'Cross entropy loss function calls '
                'tf.nn.sparse_softmax_cross_entropy_with_logits '
                'which always performs a softmax internally.')
            self._softmax = False

    def layer_op(self, prediction, ground_truth, noise, weight_map=None):
        """
        Compute loss from `prediction` and `ground truth`,
        the computed loss map are weighted by `weight_map`.

        if `prediction `is list of tensors, each element of the list
        will be compared against `ground_truth` and the weighted by
        `weight_map`. (Assuming the same gt and weight across scales)

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :param noise: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels, ...)``
        :return:
        """

        with tf.device('/cpu:0'):

            # prediction should be a list for multi-scale losses
            # single scale ``prediction`` is converted to ``[prediction]``
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):

                # go through each scale
                def _batch_i_loss(*args):
                    """
                    loss for the `b_id`-th batch (over spatial dimensions)

                    :param b_id:
                    :return:
                    """
                    # unpacking input from map_fn elements
                    if len(args[0]) == 2:
                        # pred and ground_truth
                        pred_b, ground_truth_b = args[0]
                        weight_b = None
                        noise_b = None
                    else:
                        pred_b, ground_truth_b, noise_b, weight_b = args[0]

                    pred_b = tf.reshape(pred_b, [-1, self._num_classes])

                    # performs softmax if required
                    if self._softmax:
                        pred_b = tf.cast(pred_b, dtype=tf.float32)
                        pred_b = tf.nn.softmax(pred_b)

                    # reshape pred, ground_truth, weight_map to the same
                    # size: (n_voxels, num_classes)
                    # if the ground_truth has only one channel, the shape
                    # becomes: (n_voxels,)
                    if not pred_b.shape.is_fully_defined():
                        ref_shape = tf.stack(
                            [tf.shape(pred_b)[0], tf.constant(-1)], 0)
                    else:
                        ref_shape = pred_b.shape.as_list()[:-1] + [-1]

                    ground_truth_b = tf.reshape(ground_truth_b, ref_shape)
                    if ground_truth_b.shape.as_list()[-1] == 1:
                        ground_truth_b = tf.squeeze(ground_truth_b, axis=-1)

                    if noise_b is not None:
                        noise_b = tf.reshape(noise_b, ref_shape)
                        if noise_b.shape.as_list()[-1] == 1:
                            noise_b = tf.squeeze(noise_b, axis=-1)

                    if weight_b is not None:
                        weight_b = tf.reshape(weight_b, ref_shape)
                        if weight_b.shape.as_list()[-1] == 1:
                            weight_b = tf.squeeze(weight_b, axis=-1)

                    # preparing loss function parameters
                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'noise': noise_b,
                        'weight_map': weight_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    return tf.to_float(self._data_loss_func(**loss_params))

                if weight_map is not None:
                    elements = (pred, ground_truth, noise, weight_map)
                else:
                    elements = (pred, ground_truth, noise)

                loss_batch = tf.map_fn(
                    fn=_batch_i_loss,
                    elems=elements,
                    dtype=tf.float32,
                    parallel_iterations=1)

                # loss averaged over batch
                data_loss.append(tf.reduce_mean(loss_batch))
            # loss averaged over multiple scales
            return tf.reduce_mean(data_loss)


def scaled_cross_entropy_approx(prediction, ground_truth, noise, constant=None):
    """
    Approximation of the scaled cross entropy seen in Kendall et al. CVPR 2018
    :param prediction:
    :param ground_truth:
    :param noise: network trained on to predict noise = log(sigma^2) so sigma^2 = exp(noise)
    :param constant: numerical constant to add to sigma^2
    :return:
    """
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    ground_truth = tf.cast(ground_truth, tf.int32)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=prediction, labels=ground_truth)

    if constant > 0. and constant is not None:
        noise = tf.log(tf.exp(noise) + constant)

    sigma_squared = tf.exp(noise)
    scaled_loss = tf.divide(loss, sigma_squared) + 0.5*noise
    return tf.reduce_mean(scaled_loss)


def scaled_cross_entropy(prediction, ground_truth, noise, constant=None):
    """
    Scaled Softmax i.e. p(y=c|x) = Softmax(logits/sigma)

    -log(Softmax) === Cross-Entropy loss

    :param prediction:
    :param ground_truth:
    :param noise: network trained on to predict noise = log(sigma^2) so sigma^2 = exp(noise)
    :param constant: numerical constant to add to sigma^2
    :return:
    """
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    ground_truth = tf.cast(ground_truth, tf.int32)

    if constant > 0. or constant is not None:
        noise = tf.log(tf.exp(noise) + constant)

    sigma_squared = tf.exp(noise)
    scaled_logits = tf.divide(prediction, sigma_squared)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=scaled_logits, labels=ground_truth)

    return tf.reduce_mean(entropy)


def stochastic_cross_entropy(prediction, ground_truth, noise, T=10):
    """
    Modelling logits as random draws from a Gaussian N(prediction, sigma^2) as seen in Kendall et al. NIPS 2017

    1. x | W ~ Normal(fW, sigma^2)
    2. p = Softmax(x)
    where fW is the network parameterised with weights W

    i. The loss is the expected cross-entropy (log-likelihood of Softmax) under the normal distribution N(fW, sigma^2)
    ii. N(fW, sigma^2) can be reparameterised as x | W  ~ fW + sigma^2 * N(0, 1)
    iii. The expected log-likelihood can therefore be expressed via MC integration by:
         loss = sum_over_voxels ( log ( 1/T sum_over_T ( Softmax(x) )

    :param prediction:
    :param ground_truth:
    :param noise:
    :param T
    :return:
    """
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    ground_truth = tf.cast(ground_truth, tf.int32)

    # define Normal distribution to sample from
    tfd = tfp.distributions
    # zero-meaned, unit-variance Normal distribution
    normal_dist = tfd.Normal(loc=0., scale=1.)

    # prediction is of size (batch_size, N_voxels, num_classes)
    # randomly sample across batches
    tensor_shape = prediction.shape

    stochastic_entropy = 0
    for _ in range(T):
        # broadcasting to multiply noise (batch_size, N_voxels, 1) with distribution (batch_size, N_voxels, num_classes)
        random_logits = prediction + tf.multiply(noise, normal_dist.sample(tensor_shape))
        stochastic_entropy += tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=random_logits, labels=ground_truth)
    stochastic_entropy = stochastic_entropy / T

    return tf.reduce_mean(stochastic_entropy)

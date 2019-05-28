# -*- coding: utf-8 -*-
"""
Loss functions for regression
"""
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.engine.application_factory import LossBayesianRegressionFactory
from niftynet.layer.base_layer import Layer


class LossFunction(Layer):
    def __init__(self,
                 loss_type='L2Loss',
                 loss_func_params=None,
                 name='loss_function'):

        super(LossFunction, self).__init__(name=name)

        # set loss function and function-specific additional params.
        self._data_loss_func = LossBayesianRegressionFactory.create(loss_type)
        self._loss_func_params = \
            loss_func_params if loss_func_params is not None else {}

    def layer_op(self,
                 prediction,
                 ground_truth=None,
                 weight_map=None):
        """
        Compute loss from ``prediction`` and ``ground truth``,
        the computed loss map are weighted by ``weight_map``.

        if ``prediction`` is list of tensors, each element of the list
        will be compared against ``ground_truth` and the weighted by
        ``weight_map``.

        :param prediction: input will be reshaped into
            ``(batch_size, N_voxels, num_classes)``
        :param ground_truth: input will be reshaped into
            ``(batch_size, N_voxels)``
        :param weight_map: input will be reshaped into
            ``(batch_size, N_voxels)``
        :return:
        """

        with tf.device('/cpu:0'):
            batch_size = ground_truth.shape[0].value
            ground_truth = tf.reshape(ground_truth, [batch_size, -1])
            if weight_map is not None:
                weight_map = tf.reshape(weight_map, [batch_size, -1])
            if not isinstance(prediction, (list, tuple)):
                prediction = [prediction]

            data_loss = []
            for ind, pred in enumerate(prediction):
                # go through each scale
                def _batch_i_loss(*args):

                    # go through each image in a batch
                    if len(args[0]) == 2:
                        pred_b, ground_truth_b = args[0]
                        weight_map_b = None
                    else:
                        pred_b, ground_truth_b, weight_map_b = args[0]
                    pred_b = tf.reshape(pred_b, [-1])

                    loss_params = {
                        'prediction': pred_b,
                        'ground_truth': ground_truth_b,
                        'weight_map': weight_map_b}
                    if self._loss_func_params:
                        loss_params.update(self._loss_func_params)

                    return tf.to_float(self._data_loss_func(**loss_params))

                if weight_map is not None:
                    elements = (pred, ground_truth, weight_map)
                else:
                    elements = (pred, ground_truth)

                loss_batch = tf.map_fn(
                    fn=_batch_i_loss,
                    elems=elements,
                    dtype=tf.float32,
                    parallel_iterations=1)
                data_loss.append(tf.reduce_mean(loss_batch))
            return tf.reduce_mean(data_loss)


def gaussian_likelihood_loss(prediction, ground_truth, noise, weight_map=None):
    """
    :param prediction: the current prediction of the ground truth.
    :param ground_truth: the measurement you are approximating with regression.
    :return: sum(differences squared) / 2 - Note, no square root
    """

    residuals = tf.subtract(prediction, ground_truth)
    if weight_map is not None:
        residuals = \
            tf.multiply(residuals, weight_map) / tf.reduce_sum(weight_map)
    return tf.nn.l2_loss(residuals)

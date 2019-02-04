# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.convolution import LearnedCategoricalGroupConvolutionalLayer
from niftynet.layer.annealing import gumbel_softmax_decay
from niftynet.network.base_net import BaseNet

import tensorflow as tf


class TestSFG(BaseNet):

    def __init__(self,
                 num_classes,
                 layer_scale,
                 p_init,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='leakyrelu',
                 name='TestSFG'):

        super(TestSFG, self).__init__(
            num_classes=num_classes,
            layer_scale=layer_scale,
            p_init=p_init,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        scale = self.layer_scale

        self.layers = [
            {'name': 'conv_0', 'n_features': int(16*scale), 'kernel_size': 3},
            {'name': 'conv_1', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'conv_2', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'conv_3', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'conv_4', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'conv_5', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'conv_6', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'conv_7', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'conv_8', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'conv_9', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'task_1_out', 'n_features': num_classes[0], 'kernel_size': 1},
            {'name': 'task_2_out', 'n_features': num_classes[1], 'kernel_size': 1}]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []
        cat_instances = []

        current_iter = unused_kwargs['current_iter']

        ### annealing of tau
        use_annealing = unused_kwargs['use_tau_annealing']
        max_tau = unused_kwargs['initial_tau']
        min_tau = unused_kwargs['min_temp']
        gs_anneal_r = unused_kwargs['gs_anneal_r']
        if use_annealing:
            # anneal every iter
            if not is_training:
                tau_val = min_tau
            else:
                tau_val = gumbel_softmax_decay(current_iter, gs_anneal_r, max_temp=max_tau, min_temp=min_tau)
                tau_val = tf.Print(tau_val, [tau_val])
        else:
            tau_val = max_tau

        for it, params in enumerate(self.layers[:-2]):
            if it == 0:
                conv_layer = LearnedCategoricalGroupConvolutionalLayer(
                    n_output_chns=params['n_features'],
                    kernel_size=params['kernel_size'],
                    categorical=True,
                    use_hardcat=unused_kwargs['use_hardcat'],
                    learn_cat=unused_kwargs['learn_categorical'],
                    p_init=self.p_init,
                    init_cat=unused_kwargs['init_categorical'],
                    constant_grouping=unused_kwargs['constant_grouping'],
                    group_connection=unused_kwargs['group_connection'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=params['name'])
                grouped_flow, learned_mask, d_p = conv_layer(images, tau_val, is_training)
                cat_instances.append((d_p, learned_mask))
                layer_instances.append((conv_layer, grouped_flow))
            else:
                conv_layer = LearnedCategoricalGroupConvolutionalLayer(
                    n_output_chns=params['n_features'],
                    kernel_size=params['kernel_size'],
                    categorical=True,
                    use_hardcat=unused_kwargs['use_hardcat'],
                    learn_cat=unused_kwargs['learn_categorical'],
                    p_init=self.p_init,
                    init_cat=unused_kwargs['init_categorical'],
                    constant_grouping=unused_kwargs['constant_grouping'],
                    group_connection=unused_kwargs['group_connection'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=params['name'])
                grouped_flow, learned_mask, d_p = conv_layer(grouped_flow, tau_val, is_training)
                cat_instances.append((d_p, learned_mask))
                layer_instances.append((conv_layer, grouped_flow))

        params = self.layers[-2]
        fc_layer_1 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            with_bn=False,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        task_1_output = fc_layer_1(grouped_flow[0], is_training)
        layer_instances.append((fc_layer_1, task_1_output))

        params = self.layers[-1]
        fc_layer_2 = ConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            acti_func=None,
            with_bn=False,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        task_2_output = fc_layer_2(grouped_flow[-1], is_training)
        layer_instances.append((fc_layer_2, task_2_output))

        if is_training:
            self._print(layer_instances)

        net_out = [task_1_output, task_2_output]
        cat_ps = [x[0] for x in cat_instances]
        return net_out, cat_ps

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

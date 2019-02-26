# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.fully_connected import FullyConnectedLayer

import tensorflow as tf


class vgg11net_A(BaseNet):
    """
    Implementation of VGG16-Net:
        Simonyan and Zisserman, "Very Deep Convolutional Networks for
                                 Large-Scale Image Recogntion", ICLR 2015

    - 2 penultimate FC layers replaced by Global Average Pooling
    - layer_5 -> avg pooling -> fc to softmax instead of
    - layer_5 -> max pooling -> fc -> fc -> fc to softmax

    - Conv layers: 'padding' = SAME, 'stride' = 1
    - Max pool: 'window' = 2x2, 'stride' = 2
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='VGG11Net_A'):

        super(vgg11net_A, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'layer_1', 'n_features': int(64/self.layer_scale), 'kernel_size': 3, 'repeat': 1},
            {'name': 'maxpool_1'},
            {'name': 'layer_2', 'n_features': int(128/self.layer_scale), 'kernel_size': 3, 'repeat': 1},
            {'name': 'maxpool_2'},
            {'name': 'layer_3', 'n_features': int(256/self.layer_scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_3'},
            {'name': 'layer_4', 'n_features': int(512/self.layer_scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_4'},
            {'name': 'layer_5', 'n_features': int(512/self.layer_scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'gap'},
            {'name': 'fc', 'n_features': self.num_classes}]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):

        layer_instances = []
        for layer_iter, layer in enumerate(self.layers):

            # Get layer type
            layer_type = self._get_layer_type(layer['name'])

            if 'repeat' in layer:
                repeat_conv = layer['repeat']
            else:
                repeat_conv = 1

            # first layer
            if layer_iter == 0:
                conv_layer = ConvolutionalLayer(
                    n_output_chns=layer['n_features'],
                    kernel_size=layer['kernel_size'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=layer['name'])
                flow = conv_layer(images, is_training)
                layer_instances.append((conv_layer, flow))
                repeat_conv = repeat_conv - 1

            # last layer
            if layer_iter == len(self.layers)-1:
                fc_layer = FullyConnectedLayer(
                    n_output_chns=layer['n_features'],
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                )
                flow = fc_layer(flow, is_training)
                layer_instances.append((fc_layer, flow))
                continue

            # all other
            if layer_type == 'maxpool':
                downsample_layer = DownSampleLayer(
                    kernel_size=2,
                    func='MAX',
                    stride=2)
                flow = downsample_layer(flow)
                layer_instances.append((downsample_layer, flow))

            elif layer_type == 'gap':
                # Global average across features
                with tf.name_scope('global_average_pool'):
                    flow = tf.reduce_mean(flow, axis=[1, 2])
                    # create bogus layer to work with print
                    tmp_layer = DownSampleLayer(func='AVG')
                    layer_instances.append((tmp_layer, flow))

            elif layer_type == 'layer':

                for _ in range(repeat_conv):
                    conv_layer = ConvolutionalLayer(
                        n_output_chns=layer['n_features'],
                        kernel_size=layer['kernel_size'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name=layer['name'])
                    flow = conv_layer(flow, is_training)
                    layer_instances.append((conv_layer, flow))

            elif layer_type == 'fc':

                fc_layer = FullyConnectedLayer(
                    n_output_chns=layer['n_features'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                )
                flow = fc_layer(flow. is_training)
                layer_instances.append((fc_layer, flow))

        if is_training:
            self._print(layer_instances)
            return flow
        return layer_instances[layer_id][1]

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    @staticmethod
    def _get_layer_type(layer_name):
        return layer_name.split('_')[0]
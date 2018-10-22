# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.fully_connected import FullyConnectedLayer

import tensorflow as tf


def apply_cross_stitch(input1, input2):
    """Cross-stich operation
    It takes two tensors input1, input2 of the same size,
    compute the linear combinations of the two with two different sets
    of coefficients kernel-wise, and output two tensors i.e.

    output1 = a*input1 + b*input2
    output2 = c*input1 + d*input2

    where a, b, c, d are tensors of size [channels].

    Retrieved from https://bit.ly/2PTnsOo

    Args:
        input1: Tensor of size [batch, width, height, channels]
        input2: Tensor of the same size as input1

    Returns:
        output1: Tensor of the same size as input1
        output2: Tensor of the same size as input2
    """
    input1_reshaped = tf.contrib.layers.flatten(input1)
    input2_reshaped = tf.contrib.layers.flatten(input2)
    input = tf.concat((input1_reshaped, input2_reshaped), axis=1)

    # initialize with identity matrix
    cross_stitch = tf.get_variable(
        "cross_stitch",
        shape=(input.shape[1], input.shape[1]),
        dtype=tf.float32,
        collections=['cross_stitches', tf.GraphKeys.GLOBAL_VARIABLES],
        initializer=tf.initializers.identity(),
    )
    output = tf.matmul(input, cross_stitch)

    # need to call .value to convert Dimension objects to normal value
    input1_shape = list(
        -1 if s.value is None else s.value for s in input1.shape,
    )
    input2_shape = list(
        -1 if s.value is None else s.value for s in input2.shape,
    )
    output1 = tf.reshape(
        output[:, :input1_reshaped.shape[1]], shape=input1_shape,
    )
    output2 = tf.reshape(
        output[:, input1_reshaped.shape[1]:], shape=input2_shape),

    return output1, output2


class DisjointBitaskVGG16Net(BaseNet):
    """ Two disjoint networks. Reference (https://arxiv.org/abs/1604.03539)

    Sharing done by:
        1) split just before last FC unit that is fed into loss function
        2) we use global average pooling instead of multiple FC units
            e.g. fc (1x1) --> l2 loss (age regression)
                 fc (1x2) --> binary softmax cross entropy (gender - 0/1)
    """
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='MT1_VGG16Net'):

        super(DisjointBitaskVGG16Net, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.task1_body_layers = [
            {'name': 'layer_1_1', 'n_features': 32 * 2, 'kernel_size': 3,
             'repeat': 2},
            {'name': 'maxpool_1_1'},
            {'name': 'layer_1_2', 'n_features': 64 * 2, 'kernel_size': 3,
             'repeat': 2},
            {'name': 'maxpool_1_2'},
            {'name': 'layer_3', 'n_features': 128 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'maxpool_1_3'},
            {'name': 'layer_4', 'n_features': 256 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'maxpool_1_4'},
            {'name': 'layer_1_5', 'n_features': 256 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'gap_1'},
        ]

        self.task2_body_layers = [
            {'name': 'layer_2_1', 'n_features': 32 * 2, 'kernel_size': 3,
             'repeat': 2},
            {'name': 'maxpool_2_1'},
            {'name': 'layer_2_2', 'n_features': 64 * 2, 'kernel_size': 3,
             'repeat': 2},
            {'name': 'maxpool_2_2'},
            {'name': 'layer_3', 'n_features': 128 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'maxpool_2_3'},
            {'name': 'layer_2_4', 'n_features': 256 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'maxpool_2_4'},
            {'name': 'layer_2_5', 'n_features': 256 * 2, 'kernel_size': 3,
             'repeat': 3},
            {'name': 'gap_2'},
        ]

        self.task1_layers = {
            'name': 'task_1_out', 'n_features': self.num_classes[0],
        }
        self.task2_layers = {
            'name': 'task_2_out', 'n_features': self.num_classes[1],
        }

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):

        # define separate trunks for both tasks
        with tf.variable_scope('vgg_body_1'):
            flow_1, layer_instances = self.create_network_graph(
                self.task1_body_layers, images, is_training,
            )

        with tf.variable_scope('vgg_body_2'):
            flow_2, layer_instances_2 = self.create_network_graph(
                self.task2_body_layers, images, is_training,
            )
            layer_instances.extend(layer_instances_2)

        # add task 1 output
        task1_layer = self.task1_layers
        with tf.variable_scope('task_1_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task1_layer['n_features'],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task1_out = fc_layer(flow_1)
            layer_instances.append((fc_layer, task1_out))

        # add task 2 output
        task2_layer = self.task2_layers
        with tf.variable_scope('task_2_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task2_layer['n_features'],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task2_out = fc_layer(flow_2)
            layer_instances.append((fc_layer, task2_out))

        if is_training:
            # This is here because the main application also returns
            # categoricals for more complex networks.
            categoricals = None
            self._print(layer_instances)
            return [task1_out, task2_out], categoricals

        return layer_instances[layer_id][1]

    def create_network_graph(self, layers, images, is_training):

        layer_instances = []
        for layer_iter, layer in enumerate(layers):

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

            # all other
            if layer_type == 'maxpool':
                downsample_layer = DownSampleLayer(
                    kernel_size=2,
                    func='MAX',
                    stride=2)
                flow = downsample_layer(flow)
                layer_instances.append((downsample_layer, flow))

            elif layer_type == 'gap':
                with tf.name_scope('global_average_pool'):
                    flow = tf.reduce_mean(flow, axis=[1, 2])
                    # dummy layer
                    dmy = DownSampleLayer(func='AVG')
                    layer_instances.append((dmy, flow))

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
                flow = fc_layer(flow)
                layer_instances.append((fc_layer, flow))

        return flow, layer_instances

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    @staticmethod
    def _get_layer_type(layer_name):
        return layer_name.split('_')[0]


class CrossStichVGG16Net(BaseNet):
    """ Cross-stitch Network. Reference (https://arxiv.org/abs/1604.03539)

    Sharing done by:
        1) split just before last FC unit that is fed into loss function
        2) we use global average pooling instead of multiple FC units
            e.g. fc (1x1) --> l2 loss (age regression)
                 fc (1x2) --> binary softmax cross entropy (gender - 0/1)
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='MT1_VGG16Net'):

        super(CrossStichVGG16Net, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.layers = [
            {'name': 'layer_1', 'n_features': 64*2, 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_1'},
            {'name': 'layer_2', 'n_features': 128*2, 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_2'},
            {'name': 'layer_3', 'n_features': 256*2, 'kernel_size': 3, 'repeat': 3},
            {'name': 'maxpool_3'},
            {'name': 'layer_4', 'n_features': 512*2, 'kernel_size': 3, 'repeat': 3},
            {'name': 'maxpool_4'},
            {'name': 'layer_5', 'n_features': 512*2, 'kernel_size': 3, 'repeat': 3},
            {'name': 'gap'}]

        self.task1_layers = {'name': 'task_1_out', 'n_features': self.num_classes[0]}
        self.task2_layers = {'name': 'task_2_out', 'n_features': self.num_classes[1]}

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):

        # main network graph
        with tf.variable_scope('vgg_body'):
            flow, layer_instances = self.create_main_network_graph(images, is_training)

        # add task 1 output
        task1_layer = self.task1_layers
        with tf.variable_scope('task_1_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task1_layer['n_features'],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task1_out = fc_layer(flow)
            layer_instances.append((fc_layer, task1_out))

        # add task 2 output
        task2_layer = self.task2_layers
        with tf.variable_scope('task_2_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task2_layer['n_features'],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task2_out = fc_layer(flow)
            layer_instances.append((fc_layer, task2_out))

        if is_training:
            # This is here because the main application also returns categoricals
            # for more complex networks..
            categoricals = None
            self._print(layer_instances)
            return [task1_out, task2_out], categoricals

        return layer_instances[layer_id][1]

    def create_main_network_graph(self, images, is_training):

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

            # all other
            if layer_type == 'maxpool':
                downsample_layer = DownSampleLayer(
                    kernel_size=2,
                    func='MAX',
                    stride=2)
                flow = downsample_layer(flow)
                layer_instances.append((downsample_layer, flow))

            elif layer_type == 'gap':
                with tf.name_scope('global_average_pool'):
                    flow = tf.reduce_mean(flow, axis=[1, 2])
                    # dummy layer
                    dmy = DownSampleLayer(func='AVG')
                    layer_instances.append((dmy, flow))

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
                flow = fc_layer(flow)
                layer_instances.append((fc_layer, flow))

        return flow, layer_instances

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    @staticmethod
    def _get_layer_type(layer_name):
        return layer_name.split('_')[0]
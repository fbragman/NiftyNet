# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.fully_connected import FullyConnectedLayer

import tensorflow as tf


def apply_cross_stitch(
        input1: tf.Tensor,
        input2: tf.Tensor,
):
    """Cross-stich operation.

    It takes two tensors input1, input2 of common size [batch, w, h, c],
    compute the linear combinations of the two with two different sets
    of coefficients kernel-wise, and output two tensors i.e.

    output1 = a * input1 + b * input2
    output2 = c * input1 + d * input2

    where a, b, c, d are tensors of size [1, 1, 1, channels].

    Args:
        input1: Tensor of size [batch, width, height, channels]
        input2: Tensor of the same size as input1

    Returns:
        output1: Tensor of the same size as input1
        output2: Tensor of the same size as input2
    """

    assert input1.shape.as_list() == input2.shape.as_list()

    # Stack them in the last dimension: => [batch, w, h, c, 2]
    input = tf.stack([input1, input2], axis=-1)

    # initialize channelwise mixing coeffecients of the cross-stich module
    # note the broadcastable shape: [1, 1, 1, c, 2]
    # TODO (Ryu): consider defining mix_coefficients as probabilities
    # e.g.
    # rho = tf.Variable(w_init, name='rho')
    # rho = tf.nn.softplus(rho)
    # mix_coefficients = tf.divide(
    #     rho, tf.reduce_sum(rho, axis=-1, keepdims=True),
    # )

    w_init = tf.constant(
        0.5, shape=[1, 1, 1, input1.shape[-1], 2], dtype=tf.float32,
    )
    mix_coefficients_task1 = tf.Variable(w_init, name="cross_stich_task1")
    mix_coefficients_task2 = tf.Variable(w_init, name="cross_stich_task2")

    # Compute the channelwise linear combination of input1 and input2
    output1 = tf.reduce_sum(input * mix_coefficients_task1, axis=-1)
    output2 = tf.reduce_sum(input * mix_coefficients_task2, axis=-1)
    assert output1.shape.as_list() == input1.shape.as_list()
    assert output2.shape.as_list() == input1.shape.as_list()

    return output1, output2


class CrossStichVGG16Net(BaseNet):
    """Cross-stitch Network.

    Two copies of the same architecture are defined for two tasks with
    feature sharing based on cross-stich modules in every layer. The mixing
    coefficients of cross-stich modules are currently defined as trainable
    parameters for each channel of the incoming feature maps, with no
    restrictions e.g. can be negative, and larger than 1.

    Note: we use global average pooling before the final FC layer.
    Reference: https://arxiv.org/abs/1604.03539
    """
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='MT1_VGG16Net'):

        super(CrossStichVGG16Net, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.task1_body_layers = [
            {'name': 'task_1_layer_1', 'n_features': 32, 'kernel_size': 3, 'repeat': 1},
            {'name': 'task_1_maxpool_1'},
            {'name': 'task_1_layer_2', 'n_features': 64, 'kernel_size': 3, 'repeat': 1},
            {'name': 'task_1_maxpool_2'},
            {'name': 'task_1_layer_3', 'n_features': 128, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_1_maxpool_3'},
            {'name': 'task_1_layer_4', 'n_features': 256, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_1_maxpool_4'},
            {'name': 'task_1_layer_5', 'n_features': 256, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_1_gap'},
        ]
        self.task2_body_layers = [
            {'name': 'task_2_layer_1', 'n_features': 32, 'kernel_size': 3, 'repeat': 1},
            {'name': 'task_2_maxpool_1'},
            {'name': 'task_2_layer_2', 'n_features': 64, 'kernel_size': 3, 'repeat': 1},
            {'name': 'task_2_maxpool_2'},
            {'name': 'task_2_layer_3', 'n_features': 128, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_2_maxpool_3'},
            {'name': 'task_2_layer_4', 'n_features': 256, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_2_maxpool_4'},
            {'name': 'task_2_layer_5', 'n_features': 256, 'kernel_size': 3, 'repeat': 2},
            {'name': 'task_2_gap'},
        ]
        self.task1_layers = {'name': 'task_1_out', 'n_features': self.num_classes[0]}
        self.task2_layers = {'name': 'task_2_out', 'n_features': self.num_classes[1]}

    def layer_op(
            self,
            images,
            enable_cross_stich=True,
            is_training=True,
            layer_id=-1,
            **unused_kwargs
    ):
        # get feature maps for task 1 and task 2
        with tf.variable_scope('vgg_body'):
            flow_1, flow_2, layer_instances = self.create_main_network_graph(
                images, enable_cross_stich, is_training,
            )

        # add task 1 output
        task1_layer = self.task1_layers
        with tf.variable_scope('task_1_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task1_layer['n_features'],
                with_bn=False,
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
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task2_out = fc_layer(flow_2)
            layer_instances.append((fc_layer, task2_out))

        if is_training:
            # This is here because the main application also returns categoricals
            # for more complex networks..
            categoricals = None
            self._print(layer_instances)
            return [task1_out, task2_out], categoricals

        return task1_out, task2_out

    def create_main_network_graph(
            self,
            images,
            enable_cross_stich,
            is_training,
    ):

        layer_instances = []
        for layer_iter, (layer_1, layer_2) in enumerate(
                zip(self.task1_body_layers, self.task2_body_layers),
        ):
            # Get layer type
            layer_type_1 = self._get_layer_type(layer_1['name'])
            layer_type_2 = self._get_layer_type(layer_2['name'])
            assert layer_type_1 == layer_type_2

            if 'repeat' in layer_1:
                repeat_conv = layer_1['repeat']
            else:
                repeat_conv = 1

            # first layer
            if layer_iter == 0:
                assert layer_1['n_features'] == layer_2['n_features']
                assert layer_1['kernel_size'] == layer_2['kernel_size']
                conv_layer_1 = ConvolutionalLayer(
                    n_output_chns=layer_1['n_features'],
                    kernel_size=layer_1['kernel_size'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=layer_1['name'])

                conv_layer_2 = ConvolutionalLayer(
                    n_output_chns=layer_2['n_features'],
                    kernel_size=layer_2['kernel_size'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=layer_1['name'])

                flow_1 = conv_layer_1(images, is_training)
                flow_2 = conv_layer_2(images, is_training)

                layer_instances.extend(
                    [(conv_layer_1, flow_1), (conv_layer_2, flow_2)],
                )
                repeat_conv -= 1

                if enable_cross_stich:
                    with tf.variable_scope("cross_stitch_1"):
                        flow_1, flow_2 = apply_cross_stitch(flow_1, flow_2)

            # all other
            if layer_type_1 == 'maxpool':
                downsample_layer = DownSampleLayer(
                    kernel_size=2,
                    func='MAX',
                    stride=2)
                flow_1 = downsample_layer(flow_1)
                flow_2 = downsample_layer(flow_2)
                layer_instances.extend(
                    [(downsample_layer, flow_1), (downsample_layer, flow_2)],
                )
            elif layer_type_1 == 'gap':
                with tf.name_scope('global_average_pool'):
                    flow_1 = tf.reduce_mean(flow_1, axis=[1, 2])
                    flow_2 = tf.reduce_mean(flow_2, axis=[1, 2])
                    # dummy layer
                    dmy_1 = DownSampleLayer(func='AVG')
                    dmy_2 = DownSampleLayer(func='AVG')
                    layer_instances.extend([(dmy_1, flow_1), (dmy_2, flow_2)])

            elif layer_type_1 == 'layer':
                assert layer_1['n_features'] == layer_2['n_features']
                assert layer_1['kernel_size'] == layer_2['kernel_size']

                for idx in range(repeat_conv):
                    conv_layer_1 = ConvolutionalLayer(
                        n_output_chns=layer_1['n_features'],
                        kernel_size=layer_1['kernel_size'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name=layer_1['name'],
                    )
                    conv_layer_2 = ConvolutionalLayer(
                        n_output_chns=layer_2['n_features'],
                        kernel_size=layer_2['kernel_size'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name=layer_2['name'],
                    )
                    flow_1 = conv_layer_1(flow_1, is_training)
                    flow_2 = conv_layer_2(flow_2, is_training)

                    layer_instances.extend(
                        [(conv_layer_1, flow_1), (conv_layer_2, flow_2)],
                    )
                    if enable_cross_stich:
                        with tf.variable_scope(
                                "cross_stitch_{}_{}".format(layer_iter, idx),
                        ):
                            flow_1, flow_2 = apply_cross_stitch(flow_1, flow_2)

            elif layer_type_1 == 'fc':
                fc_layer_1 = FullyConnectedLayer(
                    n_output_chns=layer_1['n_features'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                )
                flow_1 = fc_layer_1(flow_1)

                fc_layer_2 = FullyConnectedLayer(
                    n_output_chns=layer_2['n_features'],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                )
                flow_2 = fc_layer_2(flow_2)

                layer_instances.extend(
                    [(fc_layer_1, flow_1), (fc_layer_2, flow_2)],
                )

        return flow_1, flow_2, layer_instances

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    @staticmethod
    def _get_layer_type(layer_name):
        return layer_name.split('_')[2]
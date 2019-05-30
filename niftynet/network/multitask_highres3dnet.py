# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet


class MultiHeteroHighRes3DNet(BaseNet):
    """
    implementation of multi-task heteroscedastic network used in Bragman et al. Uncertainty in multi-task learning:
                                                                                 Joint representations for probabilistic
                                                                                 radiotherapy treatment planning

    note: number of layers and features modified from original paper

      ### Building blocks

    i. Main network - conv/resblocks/conv
    ii.     Prediction network
    iii.    Noise network
    """

    def __init__(self,
                 num_classes,
                 net_scale,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='MultiHeteroHighRes3DNet'):

        super(MultiHeteroHighRes3DNet, self).__init__(
            num_classes=num_classes,
            net_scale=net_scale,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.main_network_layers = [
            {'name': 'conv_0', 'n_features': 64, 'kernel_size': 3},
            {'name': 'res_1', 'n_features': 64, 'kernels': (3, 3), 'repeat': 2, 'dil': 1},
            {'name': 'res_2', 'n_features': 64, 'kernels': (3, 3), 'repeat': 2, 'dil': 2},
            {'name': 'res_3', 'n_features': 256, 'kernels': (3, 3), 'repeat': 2, 'dil': 4},
            {'name': 'conv_1', 'n_features': 512, 'kernel_size': 1}
        ]

        self.prediction_task_1 = [
            {'name': 'task_1_pred_0', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_1_pred_1', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_1_pred_out', 'n_features': num_classes, 'kernel_size': 1}
        ]

        self.noise_task_1 = [
            {'name': 'task_1_noise_0', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_1_noise_1', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_1_noise_out', 'n_features': 1, 'kernel_size': 1}
        ]

        self.prediction_task_2 = [
            {'name': 'task_2_pred_0', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_2_pred_1', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_2_pred_out', 'n_features': num_classes, 'kernel_size': 1}
        ]

        self.noise_task_2 = [
            {'name': 'task_2_noise_0', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_2_noise_1', 'n_features': 128, 'kernel_size': 3},
            {'name': 'task_2_noise_out', 'n_features': 1, 'kernel_size': 1}
        ]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)

        # Create representation network
        output_flow, layer_instances = self.create_representation_network(images, is_training)

        # Task 1 - prediction branch
        task_1_prediction_out, layer_instances = self.create_specific_branch(output_flow,
                                                                             is_training,
                                                                             layer_instances,
                                                                             self.prediction_task_1)

        # Task 1 - prediction branch
        task_2_prediction_out, layer_instances = self.create_specific_branch(output_flow,
                                                                             is_training,
                                                                             layer_instances,
                                                                             self.prediction_task_2)
        # Task 1 - noise branch
        task_1_noise_out, layer_instances = self.create_specific_branch(output_flow,
                                                                        is_training,
                                                                        layer_instances,
                                                                        self.noise_task_1)

        # Task 2 - noise branch
        task_2_noise_out, layer_instances = self.create_specific_branch(output_flow,
                                                                        is_training,
                                                                        layer_instances,
                                                                        self.noise_task_2)

        output = dict()
        output['task_1_prediction'] = task_1_prediction_out
        output['task_1_noise'] = task_1_noise_out
        output['task_2_prediction'] = task_2_prediction_out
        output['task_2_noise'] = task_2_noise_out

        # set training properties
        if is_training:
            self._print(layer_instances)
            return output
        return output

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    def create_representation_network(self, images, is_training):
        """
        Create representation network
        :param images:
        :param is_training:
        :return: output tensor
        """

        layer_instances = []

        for layer_iter, layer_param in enumerate(self.main_network_layers):

            layer_name = layer_param['name']

            # Convolutional Layer or Residual Block?
            conv_flag = False
            if layer_name.split('_')[0] == 'conv':
                conv_flag = True

            n_features = layer_param['n_features']
            kernel = layer_param['kernel_size']

            repeat = 1
            if 'repeat' in layer_param:
                repeat = layer_param['repeat']

            dilation_factor = 1
            if 'dil' in layer_param:
                dilation_factor = layer_param['dil']

            if conv_flag:
                conv_layer = ConvolutionalLayer(
                    n_output_chns=n_features,
                    kernel_size=kernel,
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=layer_name)

                if layer_iter == 0:
                    flow = conv_layer(images, is_training)
                else:
                    flow = conv_layer(flow, is_training)

                layer_instances.append((conv_layer, flow))

            else:
                with DilatedTensor(flow, dilation_factor=dilation_factor) as dilated:
                    for j in range(repeat):
                        res_block = HighResBlock(
                            n_features,
                            kernel,
                            acti_func=self.acti_func,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            name='%s_%d' % (layer_name, j))
                        dilated.tensor = res_block(dilated.tensor, is_training)
                        layer_instances.append((res_block, dilated.tensor))
                flow = dilated.tensor

        return flow, layer_instances

    def create_specific_branch(self, input_flow, is_training, layer_instances, layer_params):
        """
        Create task-specific branch using input_flow from other network
        :param input_flow:
        :param is_training:
        :param layer_instances:
        :param layer_params:
        :return: prediction for the branch
        """

        for layer_iter, layer_param in enumerate(layer_params):

            # No activation function or batch-norm in last layer
            if layer_iter == len(layer_params)-1:
                acti_func = None
                with_bn = False
            else:
                acti_func = self.acti_func
                with_bn = True

            n_features = layer_param['n_features']
            kernel = layer_param['kernel_size']
            layer_name = layer_param['name']

            conv_layer = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=kernel,
                acti_func=acti_func,
                with_bn=with_bn,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name=layer_name)

            if layer_iter == 0:
                flow = conv_layer(input_flow, is_training)
            else:
                flow = conv_layer(flow, is_training)

            layer_instances.append((conv_layer, flow))

        return flow, layer_instances


class HighResBlock(TrainableLayer):
    """
    This class define a high-resolution block with residual connections
    kernels

        - specify kernel sizes of each convolutional layer
        - e.g.: kernels=(5, 5, 5) indicate three conv layers of kernel_size 5

    with_res

        - whether to add residual connections to bypass the conv layers
    """

    def __init__(self,
                 n_output_chns,
                 kernels=(3, 3),
                 acti_func='relu',
                 w_initializer=None,
                 w_regularizer=None,
                 with_res=True,
                 name='HighResBlock'):

        super(HighResBlock, self).__init__(name=name)

        self.n_output_chns = n_output_chns
        if hasattr(kernels, "__iter__"):  # a list of layer kernel_sizes
            self.kernels = kernels
        else:  # is a single number (indicating single layer)
            self.kernels = [kernels]
        self.acti_func = acti_func
        self.with_res = with_res

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        output_tensor = input_tensor
        for (i, k) in enumerate(self.kernels):
            # create parameterised layers
            bn_op = BNLayer(regularizer=self.regularizers['w'],
                            name='bn_{}'.format(i))
            acti_op = ActiLayer(func=self.acti_func,
                                regularizer=self.regularizers['w'],
                                name='acti_{}'.format(i))
            conv_op = ConvLayer(n_output_chns=self.n_output_chns,
                                kernel_size=k,
                                stride=1,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                name='conv_{}'.format(i))
            # connect layers
            output_tensor = bn_op(output_tensor, is_training)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor

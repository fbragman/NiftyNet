# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.convolution import ConvLayer, ConvolutionalLayer
from niftynet.layer.convolution import LearnedCategoricalGroupConvolutionalLayer
from niftynet.layer.annealing import gumbel_softmax_decay
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet

import tensorflow as tf


class LearnedMTHighRes3DNet2(BaseNet):

    def __init__(self,
                 num_classes,
                 layer_scale,
                 p_init,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='LearnedMTHighRes3DNet2'):

        super(LearnedMTHighRes3DNet2, self).__init__(
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
            {'name': 'res_1', 'n_features': int(16*scale), 'kernels': (3, 3), 'repeat': 2},
            {'name': 'conv_1', 'n_features': int(32*scale), 'kernel_size': 3},
            {'name': 'res_2', 'n_features': int(32*scale), 'kernels': (3, 3), 'repeat': 2},
            {'name': 'conv_2', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'res_3', 'n_features': int(64*scale), 'kernels': (3, 3), 'repeat': 2},
            {'name': 'conv_3', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'conv_4', 'n_features': int(64*scale), 'kernel_size': 3},
            {'name': 'task_1_out', 'n_features': num_classes[0], 'kernel_size': 1},
            {'name': 'task_2_out', 'n_features': num_classes[1], 'kernel_size': 1}]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        assert layer_util.check_spatial_dims(
            images, lambda x: x % 8 == 0)
        # go through self.layers, create an instance of each layer
        # and plugin data
        layer_instances = []
        cat_instances = []

        # batch-wise sampling from Cat(p) or GS(p)
        batch_sampling = unused_kwargs['batch_sampling']
        if batch_sampling is None:
            batch_sampling = False

        # current iteration for tau annealing
        current_iter = unused_kwargs['current_iter']

        ### annealing of tau
        use_annealing = unused_kwargs['use_tau_annealing']
        max_tau = unused_kwargs['initial_tau']
        min_tau = unused_kwargs['min_temp']
        gs_anneal_r = unused_kwargs['gs_anneal_r']
        if use_annealing:
            # anneal every iter
            if not is_training:
                tau_val = 0.05
            else:
                tau_val = gumbel_softmax_decay(current_iter, gs_anneal_r, max_temp=max_tau, min_temp=min_tau)
                tau_val = tf.Print(tau_val, [tau_val])
        else:
            tau_val = max_tau

        ### first convolution layer
        params = self.layers[0]
        conv_layer = LearnedCategoricalGroupConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            categorical=True,
            use_hardcat=unused_kwargs['use_hardcat'],
            learn_cat=unused_kwargs['learn_categorical'],
            p_init=self.p_init,
            batch_sampling=batch_sampling,
            init_cat=unused_kwargs['init_categorical'],
            constant_grouping=unused_kwargs['constant_grouping'],
            group_connection=unused_kwargs['group_connection'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        grouped_flow, learned_mask, d_p = conv_layer(images, tau_val, is_training)
        layer_instances.append((conv_layer, grouped_flow))
        cat_instances.append((d_p, learned_mask))

        # Output of grouped_flow is a list: [task_1, shared, task_2]
        # where task_1 = task_1 + shared
        #       task_2 = task_2 + shared

        # Need to pass learned_mask to each successive residual block
        # If is_training=False, learned_mask <- Categorical draw
        # Pass sparse tensors to residual blocks which need sparse BN vectors

        # task_1_mask = task_1_mask + shared_mask
        # task_2_mask = task_2_mask + shared_mask
        mask_to_resblock = learned_mask
        mask_to_resblock[0] = mask_to_resblock[0] + mask_to_resblock[1]
        mask_to_resblock[2] = mask_to_resblock[2] + mask_to_resblock[1]

        ### resblocks, all kernels dilated by 1 (normal convolution) on sparse tensors
        params = self.layers[1]
        # iterate over clustered activation maps
        clustered_res_block = []
        for clustered_tensor, cluster_mask in zip(grouped_flow, mask_to_resblock):
            with DilatedTensor(clustered_tensor, dilation_factor=1) as dilated:
                for j in range(params['repeat']):
                    res_block = HighResBlock(
                        params['n_features'],
                        params['kernels'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name='%s_%d' % (params['name'], j))
                    dilated.tensor = res_block(dilated.tensor, is_training,
                                               mask=cluster_mask,
                                               batch_sampling=batch_sampling)
                    layer_instances.append((res_block, dilated.tensor))
            clustered_res_block.append(dilated.tensor)

        # Do not need diagonal connections between specific ResBlock channels [task1, shared, task2]
        # Diagonal connections happen beforehand

        params = self.layers[2]
        conv_layer = LearnedCategoricalGroupConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            categorical=True,
            use_hardcat=unused_kwargs['use_hardcat'],
            learn_cat=unused_kwargs['learn_categorical'],
            p_init=self.p_init,
            batch_sampling=batch_sampling,
            init_cat=unused_kwargs['init_categorical'],
            constant_grouping=unused_kwargs['constant_grouping'],
            group_connection=unused_kwargs['group_connection'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        grouped_flow, learned_mask, d_p = conv_layer(clustered_res_block, tau_val, is_training)
        layer_instances.append((conv_layer, grouped_flow))
        cat_instances.append((d_p, learned_mask))

        # task_1_mask = task_1_mask + shared_mask
        # task_2_mask = task_2_mask + shared_mask
        mask_to_resblock = learned_mask
        mask_to_resblock[0] = mask_to_resblock[0] + mask_to_resblock[1]
        mask_to_resblock[2] = mask_to_resblock[2] + mask_to_resblock[1]

        ### resblocks, all kernels dilated by 2
        params = self.layers[3]
        clustered_res_block = []
        for clustered_tensor, cluster_mask in zip(grouped_flow, mask_to_resblock):
            with DilatedTensor(clustered_tensor, dilation_factor=2) as dilated:
                for j in range(params['repeat']):
                    res_block = HighResBlock(
                        params['n_features'],
                        params['kernels'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name='%s_%d' % (params['name'], j))
                    dilated.tensor = res_block(dilated.tensor, is_training,
                                               mask=cluster_mask,
                                               batch_sampling=batch_sampling)
                    layer_instances.append((res_block, dilated.tensor))
            clustered_res_block.append(dilated.tensor)

        params = self.layers[4]
        conv_layer = LearnedCategoricalGroupConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            categorical=True,
            use_hardcat=unused_kwargs['use_hardcat'],
            learn_cat=unused_kwargs['learn_categorical'],
            p_init=self.p_init,
            batch_sampling=batch_sampling,
            init_cat=unused_kwargs['init_categorical'],
            constant_grouping=unused_kwargs['constant_grouping'],
            group_connection=unused_kwargs['group_connection'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        grouped_flow, learned_mask, d_p = conv_layer(clustered_res_block, tau_val, is_training)
        layer_instances.append((conv_layer, grouped_flow))
        cat_instances.append((d_p, learned_mask))

        # task_1_mask = task_1_mask + shared_mask
        # task_2_mask = task_2_mask + shared_mask
        mask_to_resblock = learned_mask
        mask_to_resblock[0] = mask_to_resblock[0] + mask_to_resblock[1]
        mask_to_resblock[2] = mask_to_resblock[2] + mask_to_resblock[1]

        ### resblocks, all kernels dilated by 4
        params = self.layers[5]
        clustered_res_block = []
        for clustered_tensor, cluster_mask in zip(grouped_flow, mask_to_resblock):
            with DilatedTensor(clustered_tensor, dilation_factor=4) as dilated:
                for j in range(params['repeat']):
                    res_block = HighResBlock(
                        params['n_features'],
                        params['kernels'],
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name='%s_%d' % (params['name'], j))
                    dilated.tensor = res_block(dilated.tensor, is_training,
                                               mask=cluster_mask,
                                               batch_sampling=batch_sampling)
                    layer_instances.append((res_block, dilated.tensor))
            clustered_res_block.append(dilated.tensor)

        params = self.layers[6]
        conv_layer = LearnedCategoricalGroupConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            categorical=True,
            use_hardcat=unused_kwargs['use_hardcat'],
            learn_cat=unused_kwargs['learn_categorical'],
            p_init=self.p_init,
            batch_sampling=batch_sampling,
            init_cat=unused_kwargs['init_categorical'],
            constant_grouping=unused_kwargs['constant_grouping'],
            group_connection=unused_kwargs['group_connection'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        grouped_flow, learned_mask, d_p = conv_layer(clustered_res_block, tau_val, is_training)
        layer_instances.append((conv_layer, grouped_flow))
        cat_instances.append((d_p, learned_mask))

        params = self.layers[7]
        conv_layer = LearnedCategoricalGroupConvolutionalLayer(
            n_output_chns=params['n_features'],
            kernel_size=params['kernel_size'],
            categorical=True,
            use_hardcat=unused_kwargs['use_hardcat'],
            learn_cat=unused_kwargs['learn_categorical'],
            p_init=self.p_init,
            batch_sampling=batch_sampling,
            init_cat=unused_kwargs['init_categorical'],
            constant_grouping=unused_kwargs['constant_grouping'],
            group_connection=unused_kwargs['group_connection'],
            acti_func=self.acti_func,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name=params['name'])
        grouped_flow, learned_mask, d_p = conv_layer(grouped_flow, tau_val, is_training)
        layer_instances.append((conv_layer, grouped_flow))
        cat_instances.append((d_p, learned_mask))

        ### 1x1x1 convolution layer
        params = self.layers[8]
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

        params = self.layers[9]
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

        net_out = [task_1_output, task_2_output]

        # set training properties
        if is_training:
            self._print(layer_instances)
            if batch_sampling:
                # The p's have been broadcasted to allow sampling independently across the batch
                # Reshape back to [n_kernels, n_class] shape
                # The p's should be tiled so can just any slice [i, :, :]
                cat_ps = [x[0][0, ...] for x in cat_instances]
            else:
                cat_ps = [x[0] for x in cat_instances]
            return net_out, cat_ps

        cat_ps = [x[0] for x in cat_instances]
        return net_out, cat_ps

    def _print(self, list_of_layers):
        for (op, _) in list_of_layers:
            print(op)


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

    def layer_op(self, input_tensor, is_training, mask=None, batch_sampling=False):
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
            if mask is None:
                output_tensor = bn_op(output_tensor, is_training)
            else:
                output_tensor = bn_op(output_tensor, is_training, kernel_mask=mask)
            output_tensor = acti_op(output_tensor)
            output_tensor = conv_op(output_tensor)
        # make residual connections
        if self.with_res:
            output_tensor = ElementwiseLayer('SUM')(output_tensor, input_tensor)
        return output_tensor

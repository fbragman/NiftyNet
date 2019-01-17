# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.gn import GNLayer
from niftynet.utilities.util_common import look_up_operations
from niftynet.layer.probability import Dirichlet, GumbelSoftmax, HardCategorical, Categorical
from niftynet.layer import group_ops

SUPPORTED_PADDING = set(['SAME', 'VALID'])


def default_w_initializer():
    def _initializer(shape, dtype, partition_info):
        stddev = np.sqrt(2.0 / np.prod(shape[:-1]))
        from tensorflow.python.ops import random_ops
        return random_ops.truncated_normal(shape, 0.0, stddev, dtype=tf.float32)
        # return tf.truncated_normal_initializer(
        #    mean=0.0, stddev=stddev, dtype=tf.float32)

    return _initializer


def default_b_initializer():
    return tf.constant_initializer(0.0)


class MTConvLayer(TrainableLayer):
    """
    ConvLayer to be used with LearnedCategoricalGroupConvolutionalLayer
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='conv'):
        super(MTConvLayer, self).__init__(name=name)

        self.padding = look_up_operations(padding.upper(), SUPPORTED_PADDING)
        self.n_output_chns = int(n_output_chns)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, task_mask=None, task_it=0):

        input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        w_full_size = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        # expand kernel size to include number of features
        w_full_size = w_full_size + (n_input_chns, self.n_output_chns)
        full_stride = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        full_dilation = layer_util.expand_spatial_params(
            self.dilation, spatial_rank)

        conv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])

        if task_mask is not None:
            # Masking of kernels by multiplication with learned categorical mask grouping
            #
            # W is [w, h, d, N]: w,h: kernel size (3x3), d: depth of kernel, N: number of kernels
            # e.g if W is 3x3x10x5 then input_depth=10 with kernel size: 3x3x10
            #                           output_depth=5 --> (3x3x10)x5 kernel
            #
            # learned mask M is of size N by (T+1) where T: number of tasks
            #
            #  e.g. M_t=1 = [0 0 1 1 0] task 1
            #       M_t=2 = [1 0 0 0 0] shared
            #       M_t=3 = [0 1 0 0 1] task 2
            #
            #       kernel n=1 is assigned to shared cluster
            #       kernel n=5 is assigned to task 2 cluster
            #
            # if learning hard mask --> binary mask
            # if learning soft mask --> kernels are weighted
            #
            # kernel_group = conv_kernel * task_mask_i
            #                (w x h x d x N) * (N x 1) by broadcasting masks/weights relevant kernels

            # Use normalised convolution

            # Convert task_mask (vector of class assignments) to same size as kernel with depth-wise matrices as matrix
            # of the class assignments
            #
            # i.e. task_mask = [0 1 1 2 0 1]
            # --> task_mask[:, :, :, 2] = [1 1 1; 1 1 1; 1 1 1] etc..
            # to allow convolution of kernel?

            conv_kernel_masked = conv_kernel * task_mask
            output_tensor = tf.nn.convolution(input=input_tensor,
                                              filter=conv_kernel_masked,
                                              strides=full_stride,
                                              dilation_rate=full_dilation,
                                              padding=self.padding,
                                              name="activation_conv")

        else:

            output_tensor = tf.nn.convolution(input=input_tensor,
                                              filter=conv_kernel,
                                              strides=full_stride,
                                              dilation_rate=full_dilation,
                                              padding=self.padding,
                                              name='conv')
        if not self.with_bias:
            return output_tensor

        # adding the bias term in normal ConvLayer, don't do it for learned grouping..
        if task_mask is not None:
            # only add it once otherwise it will be added many times if using task iterations
            bias_term = tf.get_variable(
                'b', shape=self.n_output_chns,
                initializer=self.initializers['b'],
                regularizer=self.regularizers['b'])

            output_tensor = tf.nn.bias_add(output_tensor,
                                           bias_term,
                                           name='add_bias')
        return output_tensor


class ConvLayer(TrainableLayer):
    """
    This class defines a simple convolution with an optional bias term.
    Please consider ``ConvolutionalLayer`` if batch_norm and activation
    are also used.
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='conv'):
        super(ConvLayer, self).__init__(name=name)

        self.padding = look_up_operations(padding.upper(), SUPPORTED_PADDING)
        self.n_output_chns = int(n_output_chns)
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.with_bias = with_bias

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor):

        input_shape = input_tensor.shape.as_list()
        n_input_chns = input_shape[-1]
        spatial_rank = layer_util.infer_spatial_rank(input_tensor)

        # initialize conv kernels/strides and then apply
        w_full_size = layer_util.expand_spatial_params(
            self.kernel_size, spatial_rank)
        # expand kernel size to include number of features
        w_full_size = w_full_size + (n_input_chns, self.n_output_chns)
        full_stride = layer_util.expand_spatial_params(
            self.stride, spatial_rank)
        full_dilation = layer_util.expand_spatial_params(
            self.dilation, spatial_rank)

        conv_kernel = tf.get_variable(
            'w', shape=w_full_size,
            initializer=self.initializers['w'],
            regularizer=self.regularizers['w'])

        output_tensor = tf.nn.convolution(input=input_tensor,
                                          filter=conv_kernel,
                                          strides=full_stride,
                                          dilation_rate=full_dilation,
                                          padding=self.padding,
                                          name='conv')
        if not self.with_bias:
            return output_tensor

        # adding the bias term
        bias_term = tf.get_variable(
            'b', shape=self.n_output_chns,
            initializer=self.initializers['b'],
            regularizer=self.regularizers['b'])

        output_tensor = tf.nn.bias_add(output_tensor,
                                       bias_term,
                                       name='add_bias')
        return output_tensor


class ConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 group_size=-1,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="conv"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.group_size = group_size
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.with_bn and group_size > 0:
            raise ValueError('only choose either batchnorm or groupnorm')
        if self.with_bn:
            self.layer_name += '_bn'
        if self.group_size > 0:
            self.layer_name += '_gn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(ConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):
        conv_layer = ConvLayer(n_output_chns=self.n_output_chns,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding,
                               with_bias=self.with_bias,
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               name='conv_')

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')
            bn_layer = BNLayer(
                regularizer=self.regularizers['w'],
                moving_decay=self.moving_decay,
                eps=self.eps,
                name='bn_')
        if self.group_size > 0:
            gn_layer = GNLayer(
                regularizer=self.regularizers['w'],
                group_size=self.group_size,
                eps=self.eps,
                name='gn_')
        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        def activation(output_tensor):
            if self.with_bn:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.group_size > 0:
                output_tensor = gn_layer(output_tensor)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor,
                                              keep_prob=keep_prob)
            return output_tensor

        if self.preactivation:
            output_tensor = conv_layer(activation(input_tensor))
        else:
            output_tensor = activation(conv_layer(input_tensor))

        return output_tensor


class DirichletGumbelGroupConvolutionalLayer(TrainableLayer):

    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    The grouping is learned by a hierarchical model consisting of:
        a) p ~ Dir(m, s)
        b) x ~ Cat(p) where Cat is approximated by a Gumbel-Softmax with temperature tau

    Each convolutional kernel has a separate hierarchical model
    The total uncertainty in architecture per layer is prod_i s_i

    Task Mask M of form: input_depth x n_tasks + 1 where M[:, -1] is always the shared representation

    Note: a prior should be put on Dirichlet m and s in the form of a hyper prior

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 n_tasks=2,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 temperature=1,
                 keep_group=True,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="d_conv"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.with_bn:
            self.layer_name += '_bn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(DirichletGumbelGroupConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        # for group convolution: n_tasks + 1 since task specific + shared
        self.n_tasks = n_tasks + 1
        # for group convolution: either output single tensor or multiple tensors as determined by grouping
        self.keep_group = keep_group

        # for Gumbel-Softmax
        self.tau = temperature

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')

            # Only need 1 batch-norm layer since grouping is done by masking input tensor
            group_bn = BNLayer(regularizer=self.regularizers['w'],
                               moving_decay=self.moving_decay,
                               eps=self.eps,
                               name='bn_')
        else:
            group_bn = None

        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        # Convolutional layer
        #  ConvLayer is an instance of TrainableLayer
        # __call__ method for TrainableLayer is overrided with tf.make_template
        # group_conv trainable variables are easily shared in downstream group iteration
        conv_layer = ConvLayer(n_output_chns=self.n_output_chns,
                               kernel_size=self.kernel_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding,
                               with_bias=self.with_bias,
                               w_initializer=self.initializers['w'],
                               w_regularizer=self.regularizers['w'],
                               b_initializer=self.initializers['b'],
                               b_regularizer=self.regularizers['b'],
                               name='group_conv_')

        def activation(output_tensor, bn_layer):
            if self.with_bn and bn_layer is not None:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor, keep_prob=keep_prob)
            return output_tensor

        # Create tensorflow variables for mean and precision
        # Regularisation on precision/mean --> set up hyperprior on Dirichlet distribution
        # Dirichlet parameters should be size [input_depth, ...]
        with tf.variable_scope('path_learning'):
            with tf.variable_scope('dirichlet'):

                # Depth of kernel
                kernel_depth = input_tensor.shape.as_list()[-1]

                # Create custom initializer
                precision_init = 5 * np.ones((kernel_depth,), dtype=np.float32)
                mean_init = np.ones((kernel_depth, self.n_tasks), dtype=np.float32)

                precision = tf.get_variable('precision',
                                            initializer=precision_init,
                                            dtype=tf.float32,
                                            trainable=True)

                mean = tf.get_variable('mean',
                                       initializer=mean_init,
                                       dtype=tf.float32,
                                       trainable=True)

                # Dirichlet sampling per kernel
                kernel_dirichlets = Dirichlet(mean, precision, num_samples=100)
                kernel_probs = kernel_dirichlets()

                # Categorical sampling per Dirichlet kernel
                kernel_categoricals = GumbelSoftmax(kernel_probs, self.tau)
                # Output of kernel_mask is a tensor of shape = (n_kernels, n_tasks + 1)
                kernel_mask = kernel_categoricals(hard=True)
                # Unpack
                kernal_masks_unpacked = tf.unstack(kernel_mask, axis=1)

        # Convolution across task features
        output_convs = []
        for task_mask in kernal_masks_unpacked:
            # Reshape task_mask to 1x1xc1x1 size
            # Put code
            task_mask = tf.expand_dims(task_mask, 1) * tf.ones([1, self.n_output_chns])
            task_mask = tf.expand_dims(tf.expand_dims(task_mask, 0), 0)
            if self.preactivation:
                output_convs.append(conv_layer(activation(input_tensor, group_bn), task_mask))
            else:
                output_convs.append(activation(conv_layer(input_tensor, task_mask), group_bn))

        # sum all tasks
        if self.keep_group:
            output_tensor = tf.add_n(output_convs)
        else:
            output_tensor = []
            for task_it in range(self.n_tasks-1):
                tmp_tensors = [output_convs[i] for i in [task_it, -1]]
                output_tensor.append(tf.add_n(tmp_tensors))

        return output_tensor

    def get_masking_by_grouping(self, input_tensor, dirichlet_group):

        # Create dummy mask of 1s
        mask_1 = tf.ones_like(input_tensor)
        # Create dummy mask of 0s
        mask_0 = tf.zeros_like(input_tensor)

        # Create splits
        with tf.variable_scope('tmp_masks'):
            X1_split = tf.split(value=mask_1, num_or_size_splits=dirichlet_group[0],
                                axis=-1, name='dmy_1')
            X2_split = tf.split(value=mask_0, num_or_size_splits=dirichlet_group[0],
                                axis=-1, name='dmy_0')

        # Create indexing matrices for hard-coded mask
        m1 = np.identity(3)
        m2 = 1-m1

        masked_tensors = []
        masks = []
        for itr in range(3):

            idx_m2 = np.nonzero(m2[itr, :])[0]
            tmp_X1 = [X1_split[itr]]

            if itr == 0:
                mask = tf.concat(tmp_X1 + [X2_split[idx_m2[0]]] + [X2_split[idx_m2[1]]], axis=-1)
            elif itr == 1:
                mask = tf.concat([X2_split[idx_m2[0]]] + tmp_X1 + [X2_split[idx_m2[1]]], axis=-1)
            else:
                mask = tf.concat([X2_split[idx_m2[0]]] + [X2_split[idx_m2[1]]] + tmp_X1, axis=-1)

            tmp = tf.multiply(input_tensor, mask)
            masked_tensors.append(tmp)
            masks.append(mask)

        return masked_tensors, masks

    def calculate_grouping(self, input_tensor, dirichlet_sample):

        # Input tensor depth
        c1 = tf.cast(tf.convert_to_tensor(input_tensor.get_shape().as_list()[-1:][0]), dtype=tf.float32)

        # Get amount per group and round to an integer
        c1_over_g = tf.multiply(c1, dirichlet_sample)

        # Smart-round of c1_over_g
        c1_over_g = group_ops.smartround(c1_over_g, c1)
        c1_over_g = tf.cast(c1_over_g, dtype=tf.int32)

        return c1_over_g


class GroupConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    The g parameter defines the grouping

    e.g. if input feature map / image is of size (H x W x c1)

    1) g = 1, convolution with c2 filters of size h x w x c1/g --> H x W by c2
    2) g = 2, convolution with c2 filters of size h x w x c1/g over H x W x c1/g

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 group=1,
                 group_output=False,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="gconv"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.with_bn:
            self.layer_name += '_bn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(GroupConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        # for group convolution
        self.group = group
        self.group_output = group_output

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')

            # Perform batch-norm per group so as not to override potential batch statistics
            bn_layers = []
            for it in range(self.group):
                bn_layers.append(BNLayer(regularizer=self.regularizers['w'],
                                         moving_decay=self.moving_decay,
                                         eps=self.eps,
                                         name='bn_' + str(it)))
        else:
            bn_layers = None


        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        if self.group == 1:
            raise ValueError("Group convolution cannot be called for 1 group\n")

        # Input tensor channel depth
        c1 = input_tensor.get_shape().as_list()[-1:][0]
        feature_per_group = self.n_output_chns / self.group

        if (c1 % self.group) or (self.n_output_chns % self.group):
            raise ValueError('Input feature of tensor or output features has to be factorisable by group number')

        # Groups
        group_conv_layer = []
        for idx, _ in enumerate(range(self.group)):

            tmp = ConvLayer(n_output_chns=feature_per_group,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=self.padding,
                            with_bias=self.with_bias,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            b_initializer=self.initializers['b'],
                            b_regularizer=self.regularizers['b'],
                            name='conv_group_' + str(idx))

            group_conv_layer.append(tmp)

        def activation(output_tensor, bn_layer=None):
            if self.with_bn and bn_layer is not None:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor, keep_prob=keep_prob)
            return output_tensor

        # slice up input_tensor by c1/g
        sliced_tensors = tf.split(input_tensor, num_or_size_splits=self.group, axis=-1)
        sliced_output_tensor = []

        ctr = 0
        if bn_layers is not None:
            # Batch-Norm per group
            for slice_tensor, bn_layer in zip(sliced_tensors, bn_layers):

                if self.preactivation:
                    sliced_output_tensor.append(group_conv_layer[ctr](activation(slice_tensor, bn_layer=bn_layer)))
                else:
                    sliced_output_tensor.append(activation(group_conv_layer[ctr](slice_tensor), bn_layer=bn_layer))
        else:
            # No batch norm applied
            for slice_tensor in sliced_tensors:

                if self.preactivation:
                    sliced_output_tensor.append(group_conv_layer[ctr](activation(slice_tensor)))
                else:
                    sliced_output_tensor.append(activation(group_conv_layer[ctr](slice_tensor)))


        if self.group_output:
            # stack tensors back up to form output == group 1 convolutional layer with same input/output settings
            output_tensor = tf.concat(sliced_output_tensor, axis=-1)
        else:
            # output multiple tensors corresponding to each group
            output_tensor = sliced_output_tensor

        ctr += 1

        return output_tensor


class GroupSharedConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    The g parameter defines the grouping

    In this class, group-specific and shared grouping is enforced

    | task 1 |
    | shared |
    | shared |
    | task 2 |

    ### NOT DONE
    ### NEED TO FIGURE OUT BEST WAY TO DECIDE HOW MANY JOINT FEATURES
    ### ALSO, HOW DO WE BATCH-NORM?? BN just for Joint?

    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 group=1,
                 group_output=False,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 with_bias=False,
                 with_bn=True,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="gconv"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)
        if self.with_bn:
            self.layer_name += '_bn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(GroupSharedConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        # for group convolution
        self.group = group
        self.group_output = group_output

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, is_training=None, keep_prob=None):

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')

            # Perform batch-norm on a group basis
            bn_layer = []
            for it in range(self.group):
                bn_layer.append(BNLayer(regularizer=self.regularizers['w'],
                                        moving_decay=self.moving_decay,
                                        eps=self.eps,
                                        name='bn_' + str(it)))
        else:
            bn_layer = None

        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        if self.group == 1:
            raise ValueError("Group convolution cannot be called for 1 group\n")

        # Input tensor channel depth
        c1 = input_tensor.get_shape().as_list()[-1:][0]
        feature_per_group = self.n_output_chns / self.group

        if (c1 % self.group) or (self.n_output_chns % self.group):
            raise ValueError('Input feature of tensor or output features has to be factorisable by group number')

        # Groups
        group_conv_layer = []
        for idx, _ in enumerate(range(self.group)):

            tmp = ConvLayer(n_output_chns=feature_per_group,
                            kernel_size=self.kernel_size,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=self.padding,
                            with_bias=self.with_bias,
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            b_initializer=self.initializers['b'],
                            b_regularizer=self.regularizers['b'],
                            name='conv_group_' + str(idx))

            group_conv_layer.append(tmp)

        def activation(output_tensor, bn_layer):
            if self.with_bn and bn_layer is not None:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor, keep_prob=keep_prob)
            return output_tensor

        # slice up input_tensor by c1/g
        sliced_tensors = tf.split(input_tensor, num_or_size_splits=self.group, axis=-1)
        sliced_output_tensor = []

        ctr = 0
        for slice_tensor, group_bn in zip(sliced_tensors, bn_layer):

            if self.preactivation:
                sliced_output_tensor.append(group_conv_layer[ctr](activation(slice_tensor, group_bn)))
            else:
                sliced_output_tensor.append(activation(group_conv_layer[ctr](slice_tensor), group_bn))

        if self.group_output:
            # stack tensors back up to form output == group 1 convolutional layer with same input/output settings
            output_tensor = tf.concat(sliced_output_tensor, axis=-1)
        else:
            # output multiple tensors corresponding to each group
            output_tensor = sliced_output_tensor

        ctr += 1

        return output_tensor


class LearnedCategoricalMasking(TrainableLayer):
    """
    Same as LearnedCategoricalGroupConvolutionalLayer BUT

    1)  Learn masks for routing rather than clustering kernels

    Method:

        i) Learn x ~ Cat(p)
        ii) Create masks from x
        iii) Determine

    """


class LearnedCategoricalGroupConvolutionalLayer(TrainableLayer):
    """
    This class defines a composite layer with optional components::

        convolution -> batch_norm -> activation -> dropout

    Learn how to disentangle features by learning categorical distribution with Dirichlet prior
    over convolutional kernels

    Params of interest:

    1.  categorical = True
            if set to False, priors to categorical are used as soft weights to kernels

    2.  use_hard = True
            if set to True, hard approximation in fwd pass and bwd pass on continuous GS approximation
        use_hard = False
            if set to False, soft approximation with GS in fwd and bwd pass

    3.  with_gn | group_norm on clusters
            if set to True, group normalisation will be performed
            it will be called on each separate kernel group i.e. task specific and task invariant

    4.  tau | temperature for Gumbel Softmax is categorical = True
            tau = 0 GS -> Categorical
            this should be annealed over training or if learned becomes entropy regularisation (Szgedy et al, 2015)
            if learned: GS can dynamically adjust "confidence" of proposed samples during training


    The b_initializer and b_regularizer are applied to the ConvLayer
    The w_initializer and w_regularizer are applied to the ConvLayer,
    the batch normalisation layer, and the activation layer (for 'prelu')
    """

    def __init__(self,
                 n_output_chns,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='SAME',
                 categorical=True,
                 use_hardcat=True,
                 learn_cat=True,
                 init_cat=(1/3, 1/3, 1/3),
                 p_init=False,
                 constant_grouping=False,
                 use_annealing=False,
                 group_connection='mixed',
                 with_bias=False,
                 with_bn=False,
                 with_gn=False,
                 group_size=-1,
                 acti_func=None,
                 preactivation=False,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name="conv"):

        self.acti_func = acti_func
        self.with_bn = with_bn
        self.with_gn = with_gn
        self.group_size = group_size
        self.preactivation = preactivation
        self.layer_name = '{}'.format(name)

        self.categorical = categorical
        self.use_hardcat = use_hardcat
        self.learn_cat = learn_cat
        self.p_init = p_init
        self.init_cat = init_cat
        self.constant_grouping = constant_grouping

        self.group_connection = group_connection

        self.use_annealing = use_annealing

        if self.with_bn and group_size > 0:
            raise ValueError('only choose either batchnorm or groupnorm')
        if self.with_bn:
            self.layer_name += '_bn'
        if self.group_size > 0:
            self.layer_name += '_gn'
        if self.acti_func is not None:
            self.layer_name += '_{}'.format(self.acti_func)
        super(LearnedCategoricalGroupConvolutionalLayer, self).__init__(name=self.layer_name)

        # for ConvLayer
        self.n_output_chns = n_output_chns
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.with_bias = with_bias

        # for BNLayer
        self.moving_decay = moving_decay
        self.eps = eps

        self.initializers = {
            'w': w_initializer if w_initializer else default_w_initializer(),
            'b': b_initializer if b_initializer else default_b_initializer()}

        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, input_tensor, tau, is_training=None, keep_prob=None, p_init=False):

        if self.with_bn:
            if is_training is None:
                raise ValueError('is_training argument should be '
                                 'True or False unless with_bn is False')

            # Only need 1 batch-norm layer since grouping is done by masking input tensor
            group_bn = BNLayer(regularizer=self.regularizers['w'],
                               moving_decay=self.moving_decay,
                               eps=self.eps,
                               name='bn_')
        else:
            group_bn = None

        if self.acti_func is not None:
            acti_layer = ActiLayer(
                func=self.acti_func,
                regularizer=self.regularizers['w'],
                name='acti_')

        if keep_prob is not None:
            dropout_layer = ActiLayer(func='dropout', name='dropout_')

        conv_layer = MTConvLayer(n_output_chns=self.n_output_chns,
                                 kernel_size=self.kernel_size,
                                 stride=self.stride,
                                 dilation=self.dilation,
                                 padding=self.padding,
                                 with_bias=self.with_bias,
                                 w_initializer=self.initializers['w'],
                                 w_regularizer=self.regularizers['w'],
                                 b_initializer=self.initializers['b'],
                                 b_regularizer=self.regularizers['b'],
                                 name='group_conv_')

        def activation(output_tensor, bn_layer):
            if self.with_bn and bn_layer is not None:
                output_tensor = bn_layer(output_tensor, is_training)
            if self.acti_func is not None:
                output_tensor = acti_layer(output_tensor)
            if keep_prob is not None:
                output_tensor = dropout_layer(output_tensor, keep_prob=keep_prob)
            return output_tensor

        with tf.variable_scope('categorical_p'):

            # Number of kernels
            N = self.n_output_chns

            if self.p_init:
                dirichlet = tf.distributions.Dirichlet
                alpha = tf.constant([1., 1., 1.])
                dist = dirichlet(alpha)
                dirichlet_init = tf.stop_gradient(dist.sample([N]))
            else:
                #dirichlet_init_user = np.float32(np.log(np.exp(np.asarray(self.init_cat)) - 1.0))
                dirichlet_init_user = np.float32(np.asarray(self.init_cat))
                dirichlet_init = dirichlet_init_user * np.ones((N, 3), dtype=np.float32)

            # Check if using constant grouping or learning
            if self.learn_cat:
                dirichlet_p = tf.get_variable('cat_prior',
                                              initializer=dirichlet_init,
                                              dtype=tf.float32,
                                              trainable=True)

                # For variables to be in range [0, 1] - softplus
                #dirichlet_p = tf.nn.softplus(dirichlet_p)
                #dirichlet_p = tf.divide(dirichlet_p, tf.reduce_sum(dirichlet_p, axis=1, keepdims=True))
                dirichlet_p = tf.nn.softmax(dirichlet_p, axis=1)

            else:
                dirichlet_p = tf.constant(dirichlet_init)

        if self.constant_grouping:
            # create constant grouping / no sampling at each iteration
            # mixture defined by init_cat

            constant_mask = HardCategorical(dirichlet_init_user, N)
            cat_mask = constant_mask()
            cat_mask_unstacked = tf.unstack(cat_mask, axis=1)
        else:
            if self.categorical:
                if is_training:
                    # sample x|p ~ Cat(p)

                    # Run if sampling from a categorical
                    # Can be if learning p (learn_cat=True) or constant p (learn_cat=False)
                    with tf.variable_scope('categorical_sampling'):
                        # Create object for categorical
                        cat_dist = GumbelSoftmax(dirichlet_p, tau)

                        # Sample from mask - [N by 3] either one-hot (use_hardcat=True) or soft (use_hardcat=False)
                        cat_mask = cat_dist(hard=self.use_hardcat, is_training=is_training)
                        cat_mask_unstacked = tf.unstack(cat_mask, axis=1)
                else:
                    # sample from Cat(p) - do not need approximation
                    cat_dist = Categorical(dirichlet_p)
                    cat_mask = cat_dist()
                    cat_mask_unstacked = tf.unstack(cat_mask, axis=1)

            else:
                # soft weighting using p
                with tf.variable_scope('soft_weight_masks'):
                    raise NotImplementedError

        # Convolution on clustered kernels using sampled mask

        # Check whether input_tensor is list or not
        # if list: layer > 1, if not: layer == 1
        output_layers = []

        if type(input_tensor) is not list:
            for sampled_mask in cat_mask_unstacked:
                if self.preactivation:
                    output_layers.append(conv_layer(activation(input_tensor, group_bn), sampled_mask))
                else:
                    output_layers.append(activation(conv_layer(input_tensor, sampled_mask), group_bn))
        else:
            for clustered_tensor, sampled_mask in zip(input_tensor, cat_mask_unstacked):
                if self.preactivation:
                    output_layers.append(conv_layer(activation(clustered_tensor, group_bn), sampled_mask))
                else:
                    output_layers.append(activation(conv_layer(clustered_tensor, sampled_mask), group_bn))

        # apply batch norm on sparse tensors before combination
        bn_1 = BNLayer(regularizer=self.regularizers['w'],
                       moving_decay=self.moving_decay,
                       eps=self.eps,
                       name='bn_task_1')

        bn_2 = BNLayer(regularizer=self.regularizers['w'],
                       moving_decay=self.moving_decay,
                       eps=self.eps,
                       name='bn_shared')

        bn_3 = BNLayer(regularizer=self.regularizers['w'],
                       moving_decay=self.moving_decay,
                       eps=self.eps,
                       name='bn_task_2')

        output_layers[0] = bn_1(output_layers[0], is_training, kernel_mask=cat_mask_unstacked[0])
        output_layers[1] = bn_2(output_layers[1], is_training, kernel_mask=cat_mask_unstacked[1])
        output_layers[2] = bn_3(output_layers[2], is_training, kernel_mask=cat_mask_unstacked[2])

        if self.group_connection == 'mixed' or self.group_connection is None:
            with tf.name_scope('clustered_tensor_merge'):
                # task 1 tensor
                task_1_tensor = output_layers[0] + output_layers[1]

                # task 2 tensor
                task_2_tensor = output_layers[2] + output_layers[1]

                # shared tensor
                shared_tensor = output_layers[1]

        elif self.group_connection == 'separate':
            with tf.name_scope('separate_tensor_merge'):
                # task 1 tensor
                task_1_tensor = output_layers[0]

                # task 2 tensor
                task_2_tensor = output_layers[2]

                # shared tensor
                shared_tensor = output_layers[1]

        elif self.group_connection == 'dense':
            with tf.name_scope('dense_tensor_merge'):
                # task 1 tensor
                task_1_tensor = output_layers[0] + output_layers[1] + output_layers[2]

                # task 2 tensor
                task_2_tensor = task_1_tensor

                # shared tensor
                shared_tensor = task_1_tensor

        clustered_tensors = [task_1_tensor, shared_tensor, task_2_tensor]

        return clustered_tensors, cat_mask_unstacked, dirichlet_p







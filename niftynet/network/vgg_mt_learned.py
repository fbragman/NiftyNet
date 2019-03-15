# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from six.moves import range

from niftynet.layer.convolution import LearnedCategoricalGroupConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.annealing import gumbel_softmax_decay

import tensorflow as tf


class LearnedMTVGG16Net(BaseNet):
    """
    Multi-task VGG net
    Sharing done by:
        1) split just before last FC unit that is fed into loss function
            e.g. fc (1x1) --> l2 loss (age regression)
                 fc (1x2) --> binary softmax cross entropy (gender - 0/1)

    Learn task-specific and task-invariant kernels by learning:
    1) Dirichlet p (categorical parameters)
    2) x ~ Cat(p)

    Categorical is made differentiable through Gumbel-Softmax distribution that approximates
    a categorical as temperature approaches 0

    Learning either by:
    1) Soft weighting from p
    2) Soft weighting from x ~ Gumbel-Softmax(p)
    3) Hard stochastic weighting from x ~ Gumbel-Softmax(p)

    Option also for type of network structure in group_connection parameter
    1) 'seperate' - no diagonal connections between shared and task-specific
    2) 'mixed' - diagonal connections between shared and task-specific

    """

    def __init__(self,
                 num_classes,
                 layer_scale,
                 p_init,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='LearnedMTVGG16Net'):

        super(LearnedMTVGG16Net, self).__init__(
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
            {'name': 'layer_1', 'n_features': int(64/scale), 'kernel_size': 3, 'repeat': 1},
            {'name': 'maxpool_1'},
            {'name': 'layer_2', 'n_features': int(128/scale), 'kernel_size': 3, 'repeat': 1},
            {'name': 'maxpool_2'},
            {'name': 'layer_3', 'n_features': int(256/scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_3'},
            {'name': 'layer_4', 'n_features': int(512/scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'maxpool_4'},
            {'name': 'layer_5', 'n_features': int(512/scale), 'kernel_size': 3, 'repeat': 2},
            {'name': 'gap'}]

        self.task1_layers = {'name': 'task_1_out', 'n_features': self.num_classes[0]}
        self.task2_layers = {'name': 'task_2_out', 'n_features': self.num_classes[1]}

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):

        # current_iteration
        current_iter = unused_kwargs['current_iter']
        # type of connection
        group_connection = unused_kwargs['group_connection']

        # categorical learning options
        init_cat = unused_kwargs['init_categorical']
        use_hardcat = unused_kwargs['use_hardcat']
        learn_cat = unused_kwargs['learn_categorical']
        constant_grouping = unused_kwargs['constant_grouping']

        # gumbel-softmax options
        max_tau = unused_kwargs['initial_tau']
        min_tau = unused_kwargs['min_temp']
        gs_anneal_r = unused_kwargs['gs_anneal_r']
        use_annealing = unused_kwargs['use_tau_annealing']

        # concatenation or addition
        concat_tensors = unused_kwargs['concat_tensors']

        # type of intialisation
        p_init_type = unused_kwargs['p_init_type']

        # main network graph
        with tf.variable_scope('vgg_body'):
            grouped_flow, layer_instances, cats = \
                self.create_main_network_graph(images, is_training, current_iter,
                                               group_connection, use_annealing,
                                               gs_anneal_r, max_tau,
                                               use_hardcat, learn_cat,
                                               init_cat, constant_grouping,
                                               min_tau, concat_tensors, p_init_type)

        if group_connection == 'separate':
            grouped_flow[0] = grouped_flow[0] + grouped_flow[1]
            grouped_flow[2] = grouped_flow[2] + grouped_flow[1]

        # get last layer clustering
        #last_cats = cats[-1][1]
        # merge
        #task_1_mask = last_cats[0] + last_cats[1]
        #task_2_mask = last_cats[2] + last_cats[1]

        # add task 1 output
        task1_layer = self.task1_layers
        with tf.variable_scope('task_1_fc'):
            fc_layer = FullyConnectedLayer(
                n_output_chns=task1_layer['n_features'],
                with_bn=False,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
            )
            task1_out = fc_layer(grouped_flow[0], is_training)
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
            task2_out = fc_layer(grouped_flow[-1], is_training)
            layer_instances.append((fc_layer, task2_out))

        if is_training:
            self._print(layer_instances)
            cat_ps = [x[0] for x in cats]
            return [task1_out, task2_out], cat_ps

        cat_ps = [x[0] for x in cats]
        return [task1_out, task2_out], cat_ps

    def create_main_network_graph(self, images, is_training, current_iter=None,
                                  group_connection=None, use_annealing=False, gs_anneal_r=1e-5,
                                  tau_ini=1, use_hardcat=True,
                                  learn_cat=True, init_cat=(1/3, 1/3, 1/3),
                                  constant_grouping=False,
                                  min_temp=0.05, concat_tensors=True, p_init_type='cat_probability'):

        layer_instances = []
        mask_instances = []

        # Gumbel-Softmax temperature annealing
        if use_annealing:
            # anneal every iter
            if not is_training:
                tau = 0.05
            else:
                tau = gumbel_softmax_decay(current_iter, gs_anneal_r, max_temp=1.0, min_temp=min_temp)
                tau = tf.Print(tau, [tau])
        else:
            tau = tau_ini

        for layer_iter, layer in enumerate(self.layers):

            # Get layer type
            layer_type = self._get_layer_type(layer['name'])

            if 'repeat' in layer:
                repeat_conv = layer['repeat']
            else:
                repeat_conv = 1

            # first layer
            if layer_iter == 0:
                conv_layer = LearnedCategoricalGroupConvolutionalLayer(
                    n_output_chns=layer['n_features'],
                    kernel_size=layer['kernel_size'],
                    categorical=True,
                    use_hardcat=use_hardcat,
                    learn_cat=learn_cat,
                    p_init=self.p_init,
                    init_cat=init_cat,
                    constant_grouping=constant_grouping,
                    group_connection=group_connection,
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name=layer['name'])

                grouped_flow, learned_mask, d_p = conv_layer(images, tau, is_training,
                                                             concat_tensors=concat_tensors,
                                                             p_init_type=p_init_type)

                layer_instances.append((conv_layer, grouped_flow))
                mask_instances.append((d_p, learned_mask))

                repeat_conv = repeat_conv - 1

            # all other
            if layer_type == 'maxpool':
                downsample_layer = DownSampleLayer(
                    kernel_size=2,
                    func='MAX',
                    stride=2)
                # iterate pooling over clustered
                pooled_flow = []
                for clustered_tensor in grouped_flow:
                    pooled_flow.append(downsample_layer(clustered_tensor))
                grouped_flow = pooled_flow
                layer_instances.append((downsample_layer, grouped_flow))

            elif layer_type == 'gap':
                with tf.name_scope('global_average_pool'):
                    # Perform global average pooling over cluster
                    pooled_flow = []
                    for clustered_tensor in grouped_flow:
                        pooled_flow.append(tf.reduce_mean(clustered_tensor, axis=[1, 2]))
                    grouped_flow = pooled_flow
                    # create bogus layer to work with print
                    tmp_layer = DownSampleLayer(func='AVG')
                    layer_instances.append((tmp_layer, grouped_flow))

            elif layer_type == 'layer':

                for it in range(repeat_conv):
                    conv_layer = LearnedCategoricalGroupConvolutionalLayer(
                        n_output_chns=layer['n_features'],
                        kernel_size=layer['kernel_size'],
                        categorical=True,
                        use_hardcat=use_hardcat,
                        learn_cat=learn_cat,
                        p_init=self.p_init,
                        init_cat=init_cat,
                        constant_grouping=constant_grouping,
                        group_connection=group_connection,
                        acti_func=self.acti_func,
                        w_initializer=self.initializers['w'],
                        w_regularizer=self.regularizers['w'],
                        name=layer['name'] + '_conv_{}'.format(it))
                    grouped_flow, learned_mask, d_p = conv_layer(grouped_flow, tau, is_training,
                                                                 concat_tensors=concat_tensors,
                                                                 p_init_type=p_init_type)

                    layer_instances.append((conv_layer, grouped_flow))
                    mask_instances.append((d_p, learned_mask))

            elif layer_type == 'fc':
                # not used if just doing average pooling before branching out
                #fc_layer = FullyConnectedLayer(
                #    n_output_chns=layer['n_features'],
                #    acti_func=self.acti_func,
                #    w_initializer=self.initializers['w'],
                #    w_regularizer=self.regularizers['w'],
                #)
                #flow = fc_layer(flow)
                #layer_instances.append((fc_layer, flow))
                raise NotImplementedError

        return grouped_flow, layer_instances, mask_instances

    @staticmethod
    def _print(list_of_layers):
        for (op, _) in list_of_layers:
            print(op)

    @staticmethod
    def _get_layer_type(layer_name):
        return layer_name.split('_')[0]
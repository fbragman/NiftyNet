# -*- coding: utf-8 -*-
import tensorflow as tf

import matplotlib
import matplotlib.cm

import numpy as np

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import \
    ApplicationNetFactory, InitializerFactory, OptimiserFactory
from niftynet.engine.application_variables import \
    CONSOLE, NETWORK_OUTPUT, TF_SUMMARIES

from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.sampler_balanced_v2 import BalancedSampler
from niftynet.engine.windows_aggregator_grid import GridSamplesAggregator
from niftynet.engine.windows_aggregator_resize import ResizeSamplesAggregator
from niftynet.io.image_reader import ImageReader
from niftynet.layer.crop import CropLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer

from niftynet.layer.loss_regression import LossFunction as RegLossFunction
from niftynet.layer.loss_segmentation import LossFunction as SegLossFunction
from niftynet.layer.loss_classification import LossFunction as ClassLossFunction

from niftynet.layer.probability import entropy_loss

from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.evaluation.regression_evaluator import RegressionEvaluator
from niftynet.layer.rand_elastic_deform import RandomElasticDeformationLayer

SUPPORTED_INPUT = set(['image', 'output_1', 'output_2', 'weight', 'sampler'])


class MultiTaskApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "MULTITASK"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting multi-task application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.multitask_param = None
        self.SUPPORTED_SAMPLING = {
            'uniform': (self.initialise_uniform_sampler,
                        self.initialise_grid_sampler,
                        self.initialise_grid_aggregator),
            'weighted': (self.initialise_weighted_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
            'resize': (self.initialise_resize_sampler,
                       self.initialise_resize_sampler,
                       self.initialise_resize_aggregator),
            'balanced': (self.initialise_balanced_sampler,
                         self.initialise_grid_sampler,
                         self.initialise_grid_aggregator),
        }

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.multitask_param = task_param

        # initialise input image readers
        if self.is_training:
            reader_names = ('image', 'output_1', 'output_2', 'weight', 'sampler')
        elif self.is_inference:
            # in the inference process use `image` input only
            reader_names = ('image',)
        elif self.is_evaluation:
            reader_names = ('image', 'output_1', 'output_2', 'inferred')
        else:
            tf.logging.fatal(
                'Action `%s` not supported. Expected one of %s',
                self.action, self.SUPPORTED_PHASES)
            raise ValueError
        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        self.readers = [
            ImageReader(reader_names).initialise(
                data_param, task_param, file_list) for file_list in file_lists]

        # initialise input preprocessing layers
        mean_var_normaliser = MeanVarNormalisationLayer(image_name='image') \
            if self.net_param.whitening else None
        histogram_normaliser = HistogramNormalisationLayer(
            image_name='image',
            modalities=vars(task_param).get('image'),
            model_filename=self.net_param.histogram_ref_file,
            norm_type=self.net_param.norm_type,
            cutoff=self.net_param.cutoff,
            name='hist_norm_layer') \
            if (self.net_param.histogram_ref_file and
                self.net_param.normalisation) else None

        normalisation_layers = []
        if histogram_normaliser is not None:
            normalisation_layers.append(histogram_normaliser)
        if mean_var_normaliser is not None:
            normalisation_layers.append(mean_var_normaliser)

        volume_padding_layer = []
        if self.net_param.volume_padding_size:
            volume_padding_layer.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size,
                mode=self.net_param.volume_padding_mode))

        # initialise training data augmentation layers
        augmentation_layers = []
        if self.is_training:
            train_param = self.action_param
            if train_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=train_param.random_flipping_axes))
            if train_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=train_param.scaling_percentage[0],
                    max_percentage=train_param.scaling_percentage[1]))
            if train_param.rotation_angle:
                rotation_layer = RandomRotationLayer()
                if train_param.rotation_angle:
                    rotation_layer.init_uniform_angle(
                        train_param.rotation_angle)
                augmentation_layers.append(rotation_layer)
            if train_param.do_elastic_deformation:
                spatial_rank = list(self.readers[0].spatial_ranks.values())[0]
                augmentation_layers.append(RandomElasticDeformationLayer(
                    spatial_rank=spatial_rank,
                    num_controlpoints=train_param.num_ctrl_points,
                    std_deformation_sigma=train_param.deformation_sigma,
                    proportion_to_augment=train_param.proportion_to_deform))

        # only add augmentation to first reader (not validation reader)
        self.readers[0].add_preprocessing_layers(
            volume_padding_layer + normalisation_layers + augmentation_layers)

        for reader in self.readers[1:]:
            reader.add_preprocessing_layers(
                volume_padding_layer + normalisation_layers)

    def initialise_uniform_sampler(self):
        self.sampler = [[UniformSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_weighted_sampler(self):
        self.sampler = [[WeightedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_resize_sampler(self):
        self.sampler = [[ResizeSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            shuffle=self.is_training,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_sampler(self):
        self.sampler = [[GridSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            spatial_window_size=self.action_param.spatial_window_size,
            window_border=self.action_param.border,
            smaller_final_batch_mode=self.net_param.smaller_final_batch_mode,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_balanced_sampler(self):
        self.sampler = [[BalancedSampler(
            reader=reader,
            window_sizes=self.data_param,
            batch_size=self.net_param.batch_size,
            windows_per_image=self.action_param.sample_per_volume,
            queue_length=self.net_param.queue_length) for reader in
            self.readers]]

    def initialise_grid_aggregator(self):
        self.output_decoder = GridSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

    def initialise_resize_aggregator(self):
        self.output_decoder = ResizeSamplesAggregator(
            image_reader=self.readers[0],
            output_path=self.action_param.save_seg_dir,
            window_border=self.action_param.border,
            interp_order=self.action_param.output_interp_order,
            postfix=self.action_param.output_postfix)

    def initialise_sampler(self):
        if self.is_training:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][0]()
        elif self.is_inference:
            self.SUPPORTED_SAMPLING[self.net_param.window_sampling][1]()

    def initialise_aggregator(self):
        self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]()

    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.multitask_param.num_classes,
            layer_scale=self.multitask_param.layer_scale,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        def switch_sampler(for_training):
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                return sampler.pop_batch_op()

        if self.is_training:
            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            # Current_iteration
            current_iter = tf.placeholder(dtype=tf.float32, shape=())

            image = tf.cast(data_dict['image'], tf.float32)

            # Optional arguments
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob,
                        'current_iter': current_iter,
                        'group_connection': self.multitask_param.group_connection,
                        'use_tau_annealing': self.multitask_param.use_tau_annealing,
                        'initial_tau': self.multitask_param.tau,
                        'gs_anneal_r': self.multitask_param.gs_anneal_r,
                        'learn_categorical': self.multitask_param.learn_categorical,
                        'init_categorical': self.multitask_param.init_categorical,
                        'use_hardcat': self.multitask_param.use_hardcat,
                        'constant_grouping': self.multitask_param.constant_grouping}

            # Forward pass, categoricals will be 'None' if vanilla networks are used
            # net_out: list with outputs for tasks
            # categoricals: list of tensors of shape N_l by 3 where N_l is
            #               number of features for each layer
            net_out, categoricals = self.net(image, **net_args)
            net_out_task_1 = net_out[0]
            net_out_task_2 = net_out[1]

            # Get task strings
            task_1_type = self.multitask_param.task_1_type
            task_2_type = self.multitask_param.task_2_type

            # Create optimiser object
            with tf.name_scope('Optimiser'):
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.action_param.lr)

            # Get loss function objects
            loss_func_task_1, loss_func_task_2 = self.get_loss_functions(task_1_type, task_2_type)

            # Add cropping layer if defined
            crop_layer = CropLayer(border=self.multitask_param.loss_border)
            weight_map = data_dict.get('weight', None)
            weight_map = None if weight_map is None else crop_layer(weight_map)

            # determine whether cropping is needed (1x1 label image or actual dense prediction)
            print(net_out_task_1)
            print(data_dict['output_1'])
            if len(net_out_task_1.shape) < 3:
                data_loss_task_1 = loss_func_task_1(
                    prediction=net_out_task_1,
                    ground_truth=data_dict['output_1'])
            else:
                data_loss_task_1 = loss_func_task_1(
                    prediction=crop_layer(net_out_task_1),
                    ground_truth=crop_layer(data_dict['output_1']),
                    weight_map=weight_map)

            if len(net_out_task_2.shape) < 3:
                data_loss_task_2 = loss_func_task_2(
                    prediction=net_out_task_2,
                    ground_truth=data_dict['output_2'])
            else:
                data_loss_task_2 = loss_func_task_2(
                    prediction=crop_layer(net_out_task_2),
                    ground_truth=crop_layer(data_dict['output_2']),
                    weight_map=weight_map)

            # Multi-task loss
            data_loss = data_loss_task_1 + data_loss_task_2
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            if self.net_param.decay > 0.0 and reg_losses:
                reg_loss = tf.reduce_mean(
                    [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                loss = data_loss + reg_loss
                self.output_collector_loss(outputs_collector,
                                           [reg_loss, 'W_loss'])
            else:
                loss = data_loss

            # Calculate entropy of categoricals
            if self.multitask_param.learn_categorical:
                categoricals_of_network = tf.concat(categoricals, axis=0)
                categorical_entropy = entropy_loss(categoricals_of_network)
                entropy_decay = self.multitask_param.entropy_decay

                loss = loss - (entropy_decay * categorical_entropy)

                self.output_collector_loss(outputs_collector,
                                           [entropy_decay * categorical_entropy, 'H_loss'])

            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            # collecting output variables
            self.output_collector_truck(outputs_collector,
                                        data_loss,
                                        [data_loss_task_1, data_loss_task_2],
                                        net_out,
                                        data_dict)


            # collect learned categorical parameters
            if categoricals is not None:
                self.output_collector_categoricals(outputs_collector,
                                                   categoricals)

            # collect current_iter for decay
            outputs_collector.add_to_collection(
                var=current_iter, name='current_iter',
                average_over_devices=False, collection=NETWORK_OUTPUT)

        elif self.is_inference:

            data_dict = switch_sampler(for_training=False)
            image = tf.cast(data_dict['image'], tf.float32)

            # Current_iteration
            current_iter = tf.placeholder(dtype=tf.float32, shape=())

            # Optional arguments
            net_args = {'is_training': self.is_training,
                        'keep_prob': self.net_param.keep_prob,
                        'current_iter': current_iter,
                        'group_connection': self.multitask_param.group_connection,
                        'use_tau_annealing': self.multitask_param.use_tau_annealing,
                        'initial_tau': self.multitask_param.tau,
                        'gs_anneal_r': self.multitask_param.gs_anneal_r,
                        'learn_categorical': self.multitask_param.learn_categorical,
                        'init_categorical': self.multitask_param.init_categorical,
                        'use_hardcat': self.multitask_param.use_hardcat,
                        'constant_grouping': self.multitask_param.constant_grouping}

            net_out, _ = self.net(image, **net_args)
            net_out_task_1 = net_out[0]
            net_out_task_2 = net_out[1]
            task_1_type = self.multitask_param.task_1_type
            task_2_type = self.multitask_param.task_2_type

            if task_1_type == 'regression':
                net_out_task_1 = PostProcessingLayer('IDENTITY')(net_out_task_1)
            elif task_1_type == 'classification':
                output_prob = self.multitask_param.output_prob_task_1
                num_classes = self.multitask_param.num_classes[0]
                post_process_layer = self.inference_prob_output(output_prob, num_classes)
                net_out_task_1 = post_process_layer(net_out_task_1)
            elif task_1_type == 'segmentation':
                output_prob = self.multitask_param.output_prob_task_1
                num_classes = self.multitask_param.num_classes[0]
                post_process_layer = self.inference_prob_output(output_prob, num_classes)
                net_out_task_1 = post_process_layer(net_out_task_1)

            if task_2_type == 'regression':
                net_out_task_2 = PostProcessingLayer('IDENTITY')(net_out_task_2)
            elif task_2_type == 'classification':
                output_prob = self.multitask_param.output_prob_task_2
                num_classes = self.multitask_param.num_classes[1]
                post_process_layer = self.inference_prob_output(output_prob, num_classes)
                net_out_task_2 = post_process_layer(net_out_task_2)
            elif task_2_type == 'segmentation':
                output_prob = self.multitask_param.output_prob_task_2
                num_classes = self.multitask_param.num_classes[1]
                post_process_layer = self.inference_prob_output(output_prob, num_classes)
                net_out_task_2 = post_process_layer(net_out_task_2)

            #TODO add output for learned categorical grouping
            #if categoricals is not None:
            #    self.categorical_output()

            # Collections
            outputs_collector.add_to_collection(
                var=net_out_task_1, name='task_' + task_1_type,
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=net_out_task_2, name='task_' + task_2_type,
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)

            self.initialise_aggregator()

    def categorical_output(self):
        """
        Output of categoricals...
        :return:
        """
        raise NotImplementedError

    def inference_prob_output(self, output_prob, num_classes):
        """
        Function for determinining how to output classification output
        :param output_prob:
        :param num_classes:
        :return:
        """
        if output_prob and num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'SOFTMAX', num_classes=num_classes)
        elif not output_prob and num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'ARGMAX', num_classes=num_classes)
        else:
            post_process_layer = PostProcessingLayer(
                'IDENTITY', num_classes=num_classes)
        return post_process_layer

    def get_loss_functions(self, task_1_type, task_2_type):
        """
        Output loss function objects for tasks
        :return:
        """

        if task_1_type == 'regression':
            loss_func_task_1 = RegLossFunction(loss_type=self.multitask_param.loss_task_1)
        elif task_1_type == 'segmentation':
            loss_func_task_1 = SegLossFunction(n_class=self.multitask_param.num_classes[0],
                                               loss_type=self.multitask_param.loss_task_1)
        elif task_1_type == 'classification':
            loss_func_task_1 = ClassLossFunction(n_class=self.multitask_param.num_classes[0],
                                                 loss_type=self.multitask_param.loss_task_1)

        if task_2_type == 'regression':
            loss_func_task_2 = RegLossFunction(loss_type=self.multitask_param.loss_task_2)
        elif task_2_type == 'segmentation':
            loss_func_task_2 = SegLossFunction(n_class=self.multitask_param.num_classes[1],
                                               loss_type=self.multitask_param.loss_task_2)
        elif task_2_type == 'classification':
            loss_func_task_2 = ClassLossFunction(n_class=self.multitask_param.num_classes[1],
                                                 loss_type=self.multitask_param.loss_task_2)
        return loss_func_task_1, loss_func_task_2

    def output_collector_loss(self, output_collector, loss):
        """
        Output l2 weight loss and entropy loss
        :param output_collector:
        :param loss: list [loss_tensor, loss_string]
        :return:
        """
        output_collector.add_to_collection(
            var=loss[0], name=loss[1],
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)

    def output_collector_categoricals(self, outputs_collector, cats):
        """
        Output learned categoricals as images
        :param cats:
        :return:
        """
        for idx, cat in enumerate(cats):
            # reshape cat to image format but put each kernel as a batch
            var_name = 'group_probs_layer_' + str(idx)
            tmp_cat = tf.reduce_mean(cat, axis=0)

            task_1_cat = tmp_cat[0]
            shared_cat = tmp_cat[1]
            task_2_cat = tmp_cat[2]

            task_1_name = var_name + '/task_1'
            shared_name = var_name + '/shared'
            task_2_name = var_name + '/task_2'

            outputs_collector.add_to_collection(
                var=task_1_cat, name=task_1_name,
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=shared_cat, name=shared_name,
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=task_2_cat, name=task_2_name,
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

    def output_collector_truck(self, outputs_collector, data_loss, data_losses, net_out, data_dict):
        """
        Function that defines what is output based on multi-task tasks
        :param outputs_collector:
        :param data_loss:
        :param data_losses:
        :param net_out:
        :param data_dict:
        :return:
        """

        # collecting output variables
        outputs_collector.add_to_collection(
            var=data_loss, name='multi_task_loss',
            average_over_devices=False, collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=data_loss, name='D_loss',
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)

        task_1_string = 'task_1_' + self.multitask_param.task_1_type
        task_2_string = 'task_2_' + self.multitask_param.task_2_type

        outputs_collector.add_to_collection(
            var=data_losses[0], name=task_1_string,
            average_over_devices=False, collection=CONSOLE)

        outputs_collector.add_to_collection(
            var=data_losses[0], name=task_1_string,
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)

        outputs_collector.add_to_collection(
            var=data_losses[1], name=task_2_string,
            average_over_devices=False, collection=CONSOLE)

        outputs_collector.add_to_collection(
            var=data_losses[1], name=task_2_string,
            average_over_devices=True, summary_type='scalar',
            collection=TF_SUMMARIES)

        if self.multitask_param.task_1_type == 'classification':
            self.add_classification_statistics_(outputs_collector, net_out[0],
                                                self.multitask_param.num_classes[0],
                                                data_dict['output_1'], 'task_1')

        elif self.multitask_param.task_1_type == 'regression':
            mae_loss = RegLossFunction(loss_type='MAE')
            data_loss_task_1 = mae_loss(
                prediction=net_out[0],
                ground_truth=data_dict['output_1'])

            reporting_string = 'task_1_mae'

            outputs_collector.add_to_collection(
                var=data_loss_task_1, name=reporting_string,
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=data_loss_task_1, name=reporting_string,
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

        if self.multitask_param.task_2_type == 'classification':
            self.add_classification_statistics_(outputs_collector, net_out[1],
                                                self.multitask_param.num_classes[1],
                                                data_dict['output_2'], 'task_2')

        elif self.multitask_param.task_2_type == 'regression':

            print('MAE output')

            mae_loss = RegLossFunction(loss_type='MAE')
            data_loss_task_2 = mae_loss(
                prediction=net_out[1],
                ground_truth=data_dict['output_2'])

            reporting_string = 'task_2_mae'

            outputs_collector.add_to_collection(
                var=data_loss_task_2, name=reporting_string,
                average_over_devices=False, collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=data_loss_task_2, name=reporting_string,
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)


    def interpret_output(self, batch_output):
        if self.is_inference:
            tasks = list(batch_output.keys())[0:-1]
            return self.output_decoder.decode_batch(
                {tasks[0]: batch_output[tasks[0]],
                 tasks[1]: batch_output[tasks[1]]},
                batch_output['location'])
        return True

    def initialise_evaluator(self, eval_param):
        self.eval_param = eval_param
        self.evaluator = RegressionEvaluator(self.readers[0],
                                             self.regression_param,
                                             eval_param)

    def add_inferred_output(self, data_param, task_param):
        return self.add_inferred_output_like(data_param, task_param, 'output')

    def add_classification_statistics_(self,
                                       outputs_collector,
                                       net_out,
                                       num_classes,
                                       labels,
                                       opt_string):
        """ This method defines several monitoring metrics for classification

            TP - true positives
            TN - true negatives
            FP - false positives
            FN - false negatives

            1) Accuracy - (TP + TN) / (TP + TN + FP + FN)
            2) Precision - TP / (TP + FP)
            3) Recall - TP / (TP + FN)

         """
        prediction = tf.reshape(tf.argmax(net_out, -1), [-1])

        if num_classes == 2:
            conf_mat = tf.confusion_matrix(labels=labels, predictions=prediction, num_classes=num_classes)
            conf_mat = tf.to_float(conf_mat) / float(self.net_param.batch_size)
            TP = conf_mat[1][1]
            FP = conf_mat[0][1]
            FN = conf_mat[1][0]
            accuracy = tf.trace(conf_mat)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            outputs_collector.add_to_collection(
                var=accuracy,
                name=opt_string + '_accuracy',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=accuracy,
                name=opt_string + '_accuracy',
                average_over_devices=False,
                collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=precision,
                name=opt_string + '_precision',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

            outputs_collector.add_to_collection(
                var=recall,
                name=opt_string + '_recall',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)

        else:
            conf_mat = tf.confusion_matrix(labels=labels, predictions=prediction, num_classes=num_classes)
            conf_mat = tf.to_float(conf_mat) / float(self.net_param.batch_size)
            outputs_collector.add_to_collection(
                var=conf_mat[tf.newaxis, :, :, tf.newaxis],
                name='confusion_matrix',
                average_over_devices=True, summary_type='image',
                collection=TF_SUMMARIES)

    def colorize(self, value, vmin=None, vmax=None, cmap=None):
        """
        A utility function for TensorFlow that maps a grayscale image to a matplotlib
        colormap for use with TensorBoard image summaries.
        By default it will normalize the input value to the range 0..1 before mapping
        to a grayscale colormap.
        Arguments:
          - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
            [height, width, 1].
          - vmin: the minimum value of the range used for normalization.
            (Default: value minimum)
          - vmax: the maximum value of the range used for normalization.
            (Default: value maximum)
          - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
            (Default: 'gray')
        Example usage:
        ```
        output = tf.random_uniform(shape=[256, 256, 1])
        output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
        tf.summary.image('output', output_color)
        ```

        Returns a 3D tensor of shape [height, width, 3].
        """

        # normalize
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

        # squeeze last dim if it exists
        value = tf.squeeze(value)

        # quantize
        indices = tf.to_int32(tf.round(value * 255))

        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        colors = cm(np.arange(256))[:, :3]
        colors = tf.constant(colors, dtype=tf.float32)
        value = tf.gather(colors, indices)

        return value

    def set_iteration_update(self, iteration_message):
        """
        At each iteration ``application_driver`` calls::

            output = tf.session.run(variables_to_eval, feed_dict=data_dict)

        to evaluate TF graph elements, where
        ``variables_to_eval`` and ``data_dict`` are retrieved from
        ``iteration_message.ops_to_run`` and
        ``iteration_message.data_feed_dict``
         (In addition to the variables collected by self.output_collector).

        The output of `tf.session.run(...)` will be stored at
        ``iteration_message.current_iter_output``, and can be accessed
        from ``engine.handler_network_output.OutputInterpreter``.

        override this function for more complex operations
        (such as learning rate decay) according to
        ``iteration_message.current_iter``.
        """
        if iteration_message.is_training:
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            iteration_message.data_feed_dict[self.is_validation] = True

        if self.is_training:
            iteration_message.data_feed_dict[iteration_message.ops_to_run['niftynetout']['current_iter']] = \
                iteration_message.current_iter

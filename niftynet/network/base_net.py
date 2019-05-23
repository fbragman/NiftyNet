from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.base_layer import TrainableLayer


class BaseNet(TrainableLayer):
    """
    Template for networks
    """

    def __init__(self,
                 num_classes=0,
                 net_scale=1,
                 dropout_p=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="net_template"):

        super(BaseNet, self).__init__(name=name)

        self.num_classes = num_classes
        self.acti_func = acti_func

        self.net_scale = net_scale
        self.dropout_p = dropout_p

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        tf.logging.info('using {}'.format(name))

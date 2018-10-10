# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          discriminator
Author:             deepgray
--------------------------------------------
Description:


--------------------------------------------
Date:               18-6-13
Change Activity:

--------------------------------------------
"""
import tensorflow as tf
import ops
from log_config import logger


class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation=tf.nn.leaky_relu):
        logger.info('Init Discriminator %s', name)

        self.name = name
        self.is_train = is_train
        self.norm = norm
        self.activation = activation
        self.reuse = False

    def __call__(self, input_op):
        with tf.variable_scope(self.name, reuse=self.reuse):
            d_1 = ops.conv_block(input_op, 64, 'C64', 4, 2, self.is_train,
                                 self.reuse, norm=None, activation=self.activation)
            d_2 = ops.conv_block(d_1, 128, 'C128', 4, 2, self.is_train,
                                 self.reuse, self.norm, self.activation)
            d_3 = ops.conv_block(d_2, 256, 'C256', 4, 2, self.is_train,
                                 self.reuse, self.norm, self.activation)
            d_4 = ops.conv_block(d_3, 512, 'C512', 4, 2, self.is_train,
                                 self.reuse, self.norm, self.activation)
            d_5 = ops.conv_block(d_4, 1, 'C1', 4, 1, self.is_train,
                                 self.reuse, norm=None, activation=None, bias=True)
            self.d_out = tf.reduce_mean(d_5, axis=[1, 2, 3])

            self.reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

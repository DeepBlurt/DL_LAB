# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          generator
Author:             deepgray
--------------------------------------------
Description:
param_dict还没有去掉，作为backup plan
使用tf.get_collection()似乎更加简便

--------------------------------------------
Date:               18-6-13
Change Activity:

--------------------------------------------
"""
import tensorflow as tf
import ops
from log_config import logger


class Generator(object):
    def __init__(self, image_size, block_size, norm, is_train, name='Generator', activation=tf.nn.relu):
        self.name = name
        logger.info('Init Generator %s', name)
        self.norm = norm
        self.activation = activation
        self.image_size = image_size
        self.block_size = block_size
        self.is_train = is_train
        self.reuse = False

    def __call__(self, input_op):
        # 改回了魔术方法的实现，更加简洁
        with tf.variable_scope(self.name):
            conv1 = ops.conv_block(input_op, 32, 'conv1', 7, 1, self.is_train, self.reuse, self.norm,
                                   self.activation, pad='REFLECT')
            conv2 = ops.conv_block(conv1, 64, 'conv2', 3, 2, self.is_train, self.reuse, self.norm,
                                   self.activation)
            res = ops.conv_block(conv2, 128, 'conv3', 3, 2, self.is_train, self.reuse, self.norm,
                                 self.activation)
            for i in range(self.block_size):
                res = ops.residual_block(res, 128, 'res'+str(i), self.is_train, self.reuse, self.norm)
            deconv1 = ops.deconv_block(res, 64, 'deconv1', 3, 2, self.is_train, self.reuse,
                                       self.norm, self.activation)
            deconv2 = ops.deconv_block(deconv1, 32, 'deconv2', 3, 2, self.is_train, self.reuse,
                                       self.norm, self.activation)
            self.gen = ops.conv_block(deconv2, 3, 'conv_end', 7, 1, self.is_train, self.reuse, norm=None,
                                      activation=tf.nn.tanh, pad='REFLECT')

            self.reuse = True

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

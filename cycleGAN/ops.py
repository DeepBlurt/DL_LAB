# -*- coding: utf-8 -*-

"""
--------------------------------------------
File Name:          ops.py
Author:             deepgray
--------------------------------------------
Description:


--------------------------------------------
Date:               18-5-24
Change Activity:

--------------------------------------------
"""

import tensorflow as tf


def conv2d(input_op, name, kernel_size, stride, output_channel, bias=False, pad='SAME'):
    """
    2D卷积层
    :param input_op: 输入op
    :param name: 名称
    :param kernel_size: 卷积核size
    :param stride: 步长
    :param output_channel: 输出通道
    :param pad： type of padding
    :param bias：是否含有偏置项
    :return: 输出
    """
    input_channel = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'_kernel',
                                 shape=[kernel_size[0], kernel_size[1], input_channel, output_channel],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        output = tf.nn.conv2d(input_op, kernel, (1, stride[0], stride[1], 1), padding=pad)

        if bias:
            b_0 = tf.constant(0.0, shape=[output_channel], dtype=tf.float32)
            bias = tf.Variable(b_0, trainable=True, name='_bias')
            output = tf.nn.bias_add(output, bias)

        return output


def fc_layer(input_op, name, output_dim, activation=tf.nn.relu_layer):
    """
    全连接层
    :param input_op: 输入op
    :param name: 名称
    :param output_dim: 输出维度
    :param activation: 激活函数类型
    :return: 本层输出
    """
    input_dim = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.get_variable(scope+"_weights",
                                  shape=[input_dim, output_dim],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.Variable(tf.constant(0.0, shape=[output_dim], dtype=tf.float32),
                           trainable=True,
                           name="_bias",)

        return activation(input_op, weights, bias, name=scope)


def conv2d_transpose(input_op, name, channels, filter_size, stride, pad='SAME'):
    """
    反卷积层，没有包含bias和激活函数
    :param input_op: 输入op
    :param channels:
    :param filter_size:
    :param stride:
    :param name:
    :param pad:
    :return:
    """
    assert pad == 'SAME'
    batch, height, width, input_channels = input_op.get_shape().as_list()

    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], channels, input_channels]
    output_shape = [batch, height * stride, width * stride, channels]

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'kernel', filter_shape, dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0.0, 0.02))
        deconv = tf.nn.conv2d_transpose(input_op, kernel, output_shape, stride_shape, pad)
        return deconv


def _norm(input_op, is_train, reuse=True, norm=None):
    """
    normalization of input op, include instance normalization, batch normalization
    :param input_op:
    :param is_train:
    :param reuse:
    :param norm: class of normalization
    :return:
    """
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            # Calculate mean and variance of ops
            mean, sigma = tf.nn.moments(input_op, [1, 2], keep_dims=True)
            normalized = (input_op - mean) / (tf.sqrt(sigma) + eps)
            out = normalized
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.contrib.layers.batch_norm(input_op,
                                               decay=0.99, center=True,
                                               scale=True, is_training=is_train,
                                               updates_collections=None)
    else:
        out = input_op

    return out


def residual_block(input_op, num_filters, name, is_train, reuse, norm=False, pad='REFLECT'):
    """
    residual block implementation
    :param input_op:
    :param num_filters: number of filters
    :param name:
    :param is_train:
    :param reuse:
    :param norm:
    :param pad:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse) as scope:
            out = conv2d(input_op, scope, num_filters, (3, 3), 1, False, pad)
            out = _norm(out, is_train, reuse, norm)
            out = tf.nn.relu(out)
        with tf.variable_scope('res2', reuse=reuse) as scope:
            out = conv2d(out, scope, num_filters, (3, 3), 1, False, pad)
            out = _norm(out, is_train, reuse, norm)

        return tf.nn.relu(input_op + out)


def deconv_block(input_op, num_filters, name, k_size, stride, is_train, reuse, norm, activation):
    """
    反卷积层
    :param input_op: 输入op
    :param num_filters: 特征图个数
    :param name: 层名称
    :param k_size: 滤波器尺寸
    :param stride: 步长
    :param is_train:
    :param reuse: 设置重用
    :param norm: 正则化类型
    :param activation: 激活函数
    :return:
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        out = conv2d_transpose(input_op, scope, num_filters, k_size, stride)
        out = _norm(out, is_train, reuse, norm)
        out = activation(out, activation)
        return out


def conv_block(input_op, num_filters, name, k_size, stride, is_train, reuse, norm, activation,
               pad='SAME', bias=False):
    """
    卷积层与正则化层
    :param input_op:
    :param num_filters:
    :param name:
    :param k_size:
    :param stride:
    :param is_train:
    :param reuse:
    :param norm:
    :param activation:
    :param pad:
    :param bias:
    :return:
    """
    with tf.variable_scope(name, reuse=reuse) as scope:
        out = conv2d(input_op, scope, num_filters, (k_size, k_size), stride, bias, pad)
        out = _norm(out, is_train, reuse, norm)
        out = activation(out)
        return out

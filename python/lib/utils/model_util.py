#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: popfido
# @Date  : 2019/1/15
"""This module contains some model build related utility functions"""
import math
import tensorflow as tf

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops_v2 as init_ops
from tensorflow.python.util import nest
# from tensorflow.python.keras.optimizer_v2 import adagrad
# from tensorflow.python.keras.optimizer_v2 import adam
# from tensorflow.python.keras.optimizer_v2 import ftrl
# from tensorflow.python.keras.optimizer_v2 import gradient_descent
# from tensorflow.python.keras.optimizer_v2 import adamax
# from tensorflow.python.keras.optimizer_v2 import rmsprop
# from tensorflow.python.keras.optimizer_v2 import nadam

from typing import Callable, Optional, List

# Methods related to optimizers used in canned_estimators."""
_OPTIMIZER_CLS_NAMES = dict(
  Adadelta=tf.train.AdadeltaOptimizer,
  Adagrad=tf.train.AdagradOptimizer,
  Adam=tf.train.AdamOptimizer,
  Ftrl=tf.train.FtrlOptimizer,
  Momentum=tf.train.MomentumOptimizer,
  RMSProp=tf.train.RMSPropOptimizer,
  SGD=tf.train.GradientDescentOptimizer,
)

_ACTIVATION_FN_NAMES = dict(
    sigmoid=tf.nn.sigmoid,
    tanh=tf.nn.tanh,
    relu=tf.nn.relu,
    relu6=tf.nn.relu6,
    leaky_relu=tf.nn.leaky_relu,
    crelu=tf.nn.crelu,
    elu=tf.nn.elu,
    selu=tf.nn.selu,
    softplus=tf.nn.softplus,
    softsign=tf.nn.softsign,
)

_INITIALIZER_NAMES = dict(
    tnormal=tf.truncated_normal_initializer,
    uniform=tf.random_uniform_initializer,
    normal=tf.random_normal_initializer,
)


def add_layer_summary(value, tag) -> None:
    tf.summary.scalar('%s/fraction_of_zero_values' % tag, tf.nn.zero_fraction(value))
    tf.summary.histogram('%s/activation' % tag, value)


def check_no_sync_replicas_optimizer(optimizer) -> None:
    if isinstance(optimizer, tf.train.SyncReplicasOptimizer):
        raise ValueError(
            'SyncReplicasOptimizer does not support multi optimizers case. '
            'Therefore, it is not supported in DNNLinearCombined model. '
            'If you want to use this optimizer, please use either DNN or Linear model.')


def compute_fraction_of_zero(variables: List):
    """Given a linear variables list, compute the fraction of zero weights.

    Args:
        variables: A list or list of list of variables

    Returns:
        The fraction of zeros (sparsity) in the linear model.
    """
    with ops.name_scope('zero_fraction'):
        variables = nest.flatten(variables)

        with ops.name_scope('total_size'):
            sizes = [array_ops.size(x, out_type=dtypes.int64) for x in variables]
            total_size_int64 = math_ops.add_n(sizes)
        with ops.name_scope('total_zero'):
            total_zero_float32 = math_ops.add_n([
                control_flow_ops.cond(
                    math_ops.equal(size, constant_op.constant(0, dtype=dtypes.int64)),
                    true_fn=lambda: constant_op.constant(0, dtype=dtypes.float32),
                    false_fn=lambda: nn.zero_fraction(x) * tf.cast(size, dtypes.float32),
                    name='zero_count')
                for x, size in zip(variables, sizes)
            ])

        with ops.name_scope('compute'):
            total_size_float32 = math_ops.cast(
                total_size_int64, dtype=dtypes.float32, name='float32_size')
            zero_fraction_or_nan = total_zero_float32 / total_size_float32

        zero_fraction_or_nan = array_ops.identity(
            zero_fraction_or_nan, name='zero_fraction_or_nan')
        return zero_fraction_or_nan


def _get_activation_fn(opt) -> Callable:
    """Returns an activation function.

    Args:
        opt: string
        Supported 10 strings:
        * 'sigmoid': Returns `tf.sigmoid`.
        * 'tanh': Returns `tf.tanh`.
        * 'relu': Returns `tf.nn.relu`.
        * 'relu6': Returns `tf.nn.relu6`.
        * 'leaky_relu': Returns `tf.nn.leaky_relu`.
        * 'crelu': Returns `tf.nn.crelu`.
        * 'elu': Returns `tf.nn.elu`.
        * 'selu': Returns `tf.nn.selu`.
        * 'softplus': Returns `tf.nn.softplus`.
        * 'softsign': Returns `tf.nn.softsign`.

    Returns:
        Callable tf.nn.[opt] activation function

    Raises:
        ValueError: if `opt` is an unsupported string.
        ValueError: If `opt` is none of the above types.
    """
    if opt in _ACTIVATION_FN_NAMES.keys():
        return _ACTIVATION_FN_NAMES[opt]
    raise ValueError('Unsupported activation name: {}. Supported names are: {}'.format(
        opt, list(_ACTIVATION_FN_NAMES)))


def _get_optimizer_instance(
    opt: str,
    learning_rate: Optional[float] = 0.001,
    **kwargs
) -> tf.python.train.Optimizer:
    """Returns an optimizer instance.

    Supports the following types for the given `opt`:
        * An `Optimizer` instance string: Returns the given `opt`.
        * A supported string: Creates an `Optimizer` subclass with the given `learning_rate`.
    Supported strings:
        * 'Adagrad': Returns an `AdagradOptimizer`.
        * 'Adam': Returns an `AdamOptimizer`.
        * 'Ftrl': Returns an `FtrlOptimizer`.
        * 'RMSProp': Returns an `RMSPropOptimizer`.
        * 'SGD': Returns a `GradientDescentOptimizer`.

    Args:
        opt: An `Optimizer` instance, or supported string, as discussed above.
        learning_rate: A float. Only used if `opt` is a supported string.

    Returns:
        An `Optimizer` instance.

    Raises:
        ValueError: If `opt` is an unsupported string.
        ValueError: If `opt` is a supported string but `learning_rate` was not specified.
        ValueError: If `opt` is none of the above types.
    """
    if isinstance(opt, str):
        if opt in _OPTIMIZER_CLS_NAMES.keys():
            params = {}
            name = _OPTIMIZER_CLS_NAMES[opt]
            if name == "Ftrl":
                params["l1_regularization_strength"] = kwargs.get(
                    "l1_regularization_strength", 0.0
                )
                params["l2_regularization_strength"] = kwargs.get(
                    "l2_regularization_strength", 0.0
                )
            elif name == "Momentum" or name == "RMSProp":
                params["momentum"] = kwargs.get("momentum", 0.0)

            if learning_rate is None:
                raise ValueError('learning_rate must be specified when opt is supported string.')
            return _OPTIMIZER_CLS_NAMES[opt](learning_rate=learning_rate, **params)
        else:
            try:
                opt = eval(opt)  # eval('tf.nn.relu') tf.nn.relu
                if not isinstance(opt, tf.train.Optimizer):
                    raise ValueError('The given object is not an Optimizer instance. Given: {}'.format(opt))
                return opt
            except (AttributeError, NameError):
                raise ValueError('Unsupported optimizer option: `{}`. '
                                 'Supported names are: {} or an `Optimizer` instance.'.format(
                                    opt, list(_OPTIMIZER_CLS_NAMES)))
    else:
        raise ValueError('Unsupported optimizer option: `{}`. '
                         'Supported names are: {} or an `Optimizer` instance.'.format(
                            opt, list(_OPTIMIZER_CLS_NAMES.keys())))


def _get_initializer(init_method: str, seed: Optional[int] = 42,**param):
    if init_method == "tnormal":
        return tf.truncated_normal_initializer(
            stddev=param['init_value'], seed=seed
        )
    elif init_method == "uniform":
        return tf.random_uniform_initializer(
            -param['init_value'], param['init_value'], seed=seed
        )
    elif init_method == "normal":
        return tf.random_normal_initializer(
            stddev=param['init_value'], seed=seed
        )
    elif init_method == "xavier_normal":
        return tf.contrib.layers.xavier_initializer(uniform=False, seed=seed)
    elif init_method == "xavier_uniform":
        return tf.contrib.layers.xavier_initializer(uniform=True, seed=seed)
    elif init_method == "he_normal":
        return tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode="FAN_IN", uniform=False, seed=seed
        )
    elif init_method == "he_uniform":
        return tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode="FAN_IN", uniform=True, seed=seed
        )
    else:
        return tf.truncated_normal_initializer(
            stddev=param['init_value'], seed=seed
        )


def _get_linear_learning_rate(num_linear_feature_columns: int) -> float:
    """Returns the default learning rate of the linear model.

    The calculation is a historical artifact of this initial implementation, but
    has proven a reasonable choice.

    Args:
        num_linear_feature_columns: The number of feature columns of the linear model.

    Returns:
        A float.
    """
    default_learning_rate = 1. / math.sqrt(num_linear_feature_columns)
    return min(0.005, default_learning_rate)
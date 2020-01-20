#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""This module is based on tf.estimator.LinearClassifier.
linear logits builder for wide part"""
# TODO: add FM as linear part
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.summary import summary
from tensorflow.python.ops import variables as variable_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops

from lib.utils.model_util import compute_fraction_of_zero


def _get_expanded_variable_list(var_list):
    """Given an iterable of variables, expands them if they are partitioned.

    Args:
        var_list: An iterable of variables.

    Returns:
        A list of variables where each partitioned variable is expanded to its
        components.
    """
    returned_list = []
    for variable in var_list:
        if (isinstance(variable, variable_ops.Variable) or
                resource_variable_ops.is_resource_variable(variable) or
                isinstance(variable, ops.Tensor)):
            returned_list.append(variable)  # Single variable/tensor case.
        else:  # Must be a PartitionedVariable, so convert into a list.
            returned_list.extend(list(variable))
    return returned_list


def linear_logit_fn_builder(units, feature_columns, sparse_combiner='sum'):
    """Function builder for a linear logit_fn.

    Args:
      units: An int indicating the dimension of the logit layer.
      feature_columns: An iterable containing all the feature columns used by the model.
      sparse_combiner: A string specifying how to reduce if a categorical column
        is multivalent.  One of "mean", "sqrtn", and "sum".

    Returns:
      A logit_fn (see below).
    """

    def linear_logit_fn(features):
        """Linear model logit_fn.

        Args:
          features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.

        Returns:
          A `Tensor` representing the logits.
        """
        if feature_column_lib.is_feature_column_v2(feature_columns):
            linear_model = feature_column_lib.LinearModel(
                feature_columns=feature_columns,
                units=units,
                sparse_combiner=sparse_combiner,
                name='linear_model')
            logits = linear_model(features)
            bias = linear_model.bias

            # We'd like to get all the non-bias variables associated with this
            # LinearModel.
            # TODO: Figure out how to get shared embedding weights variable here.
            variables = linear_model.variables
            variables.remove(bias)

            # Expand (potential) Partitioned variables
            bias = _get_expanded_variable_list([bias])
        else:
            linear_model = feature_column._LinearModel(  # pylint: disable=protected-access
                feature_columns=feature_columns,
                units=units,
                sparse_combiner=sparse_combiner,
                name='linear_model')
            logits = linear_model(features)
            cols_to_vars = linear_model.cols_to_vars()
            bias = cols_to_vars.pop('bias')
            variables = cols_to_vars.values()
        variables = _get_expanded_variable_list(variables)

        if units > 1:
            summary.histogram('bias', bias)
        else:
            # If units == 1, the bias value is a length-1 list of a scalar Tensor,
            # so we should provide a scalar summary.
            summary.scalar('bias', bias[0][0])
        summary.scalar('fraction_of_zero_weights', compute_fraction_of_zero(variables))
        return logits

    return linear_logit_fn

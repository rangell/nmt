# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implmentations of custom resuable attention mechanisms."""

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import nn_ops


__all__ = ["ReusableLuongAttention", "ReusableBahdanauAttention"]


class ReusableLuongAttention(tf.contrib.seq2seq.LuongAttention):
  """Implements Luong-style (multiplicative) attention scoring.

  This attention has two forms.  The first is standard Luong attention,
  as described in:

  Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
  "Effective Approaches to Attention-based Neural Machine Translation."
  EMNLP 2015.  https://arxiv.org/abs/1508.04025

  The second is the scaled form inspired partly by the normalized form of
  Bahdanau attention.

  To enable the second form, construct the object with parameter
  `scale=True`.

  To reuse weights of a previous attention mechanism with the same name,
  construct the object with parameter `reuse=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="LuongAttention",
               reuse=False):
    """Construct the AttentionMechanism mechanism.

    Args:
      num_units: The depth of the attention mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      scale: Python boolean.  Whether to scale the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is `tf.nn.softmax`. Other options include
        `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional) The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the memory layer of the attention mechanism.
      name: Name to use when creating ops.
      reuse: Boolean, whether to reuse the weights of a previous layer by the
        same name.
    """
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(tf.contrib.seq2seq.LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype,
            _reuse=reuse),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name


class ReusableBahdanauAttention(tf.contrib.seq2seq.BahdanauAttention):
  """Implements Bahdanau-style (additive) attention.

  This attention has two forms.  The first is Bahdanau attention,
  as described in:

  Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  ICLR 2015. https://arxiv.org/abs/1409.0473

  The second is the normalized form.  This form is inspired by the
  weight normalization article:

  Tim Salimans, Diederik P. Kingma.
  "Weight Normalization: A Simple Reparameterization to Accelerate
   Training of Deep Neural Networks."
  https://arxiv.org/abs/1602.07868

  To enable the second form, construct the object with parameter
  `normalize=True`.

  To reuse weights of a previous attention mechanism with the same name,
  construct the object with parameter `reuse=True`.
  """

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="BahdanauAttention",
               reuse=False):
    """Construct the Attention mechanism.

    Args:
      num_units: The depth of the query mechanism.
      memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
      memory_sequence_length (optional): Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
      normalize: Python boolean.  Whether to normalize the energy term.
      probability_fn: (optional) A `callable`.  Converts the score to
        probabilities.  The default is `tf.nn.softmax`. Other options include
        `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
        Its signature should be: `probabilities = probability_fn(score)`.
      score_mask_value: (optional): The mask value for score before passing into
        `probability_fn`. The default is -inf. Only used if
        `memory_sequence_length` is not None.
      dtype: The data type for the query and memory layers of the attention
        mechanism.
      name: Name to use when creating ops.
      reuse: Boolean, whether to reuse the weights of a previous layer by the
        same name.
    """
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(tf.contrib.seq2seq.BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype,
            _reuse=reuse),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype,
            _reuse=reuse),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

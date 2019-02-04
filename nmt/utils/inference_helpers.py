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

import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.framework import ops

class GreedyInferenceHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  """A helper for use during inference.
  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_emb_inputs, end_token):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_emb_inputs: `float32` tensor shaped `[batch_size, 1]`, the
        embeddings of start tokens (styles in our case).
      end_token: `int32` scalar, the token that marks end of decoding.
    Raises:
      ValueError: if `end_token` is not a scalar.
    """
    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    self._batch_size = tf.shape(start_emb_inputs)[0]
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")
    self._start_inputs = start_emb_inputs


class  SampleInferenceHelper(GreedyInferenceHelper):
  """A helper for use during inference.
  Uses sampling (from a distribution) instead of argmax and passes the
  result through an embedding layer to get the next input.
  """

  def __init__(self, embedding, start_emb_inputs, end_token,
               softmax_temperature=None, seed=None):
    """Initializer.
    Args:
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`. The returned tensor
        will be passed to the decoder input.
      start_emb_inputs: `float32` tensor shaped `[batch_size, 1]`, the
        embeddings of start tokens (styles in our case).
      end_token: `int32` scalar, the token that marks end of decoding.
      softmax_temperature: (Optional) `float32` scalar, value to divide the
        logits by before computing the softmax. Larger values (above 1.0) result
        in more random samples, while smaller values push the sampling
        distribution towards the argmax. Must be strictly greater than 0.
        Defaults to 1.0.
      seed: (Optional) The sampling seed.
    Raises:
      ValueError: if `end_token` is not a scalar.
    """
    super(SampleInferenceHelper, self).__init__(
        embedding, start_emb_inputs, end_token)
    self._softmax_temperature = softmax_temperature
    self._seed = seed

  def sample(self, time, outputs, state, name=None):
    """sample for SampleEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, we sample instead of argmax (greedy).
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    if self._softmax_temperature == 0.0:
      return super(SampleInferenceHelper, self).sample(time, outputs, state)

    if self._softmax_temperature is None:
      logits = outputs
    else:
      logits = outputs / self._softmax_temperature

    sample_id_sampler = categorical.Categorical(logits=logits)
    sample_ids = sample_id_sampler.sample(seed=self._seed)

    return sample_ids


class BeamSearchInferenceDecoder(tf.contrib.seq2seq.BeamSearchDecoder):
  """
    BeamSearch sampling decoder.
    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `AttentionWrapper`, then you must ensure that:
    - The encoder output has been tiled to `beam_width` via
      `tf.contrib.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `zero_state` method of this
      wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `zero_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.
    An example:
    ```
    tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.zero_state(
        dtype, batch_size=true_batch_size * beam_width)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```
    Meanwhile, with `AttentionWrapper`, coverage penalty is suggested to use
    when computing scores(https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the translation to cover all inputs.
  """
  def __init__(self,
               cell,
               embedding,
               start_emb_inputs,
               end_token,
               initial_state,
               beam_width,
               output_layer=None,
               length_penalty_weight=0.0,
               coverage_penalty_weight=0.0,
               reorder_tensor_arrays=True):
    """Initialize the BeamSearchDecoder.
    Args:
      cell: An `RNNCell` instance.
      embedding: A callable that takes a vector tensor of `ids` (argmax ids),
        or the `params` argument for `embedding_lookup`.
      start_emb_inputs: `float32` tensor shaped `[batch_size, 1]`, the
        embeddings of start tokens (styles in our case).
      end_token: `int32` scalar, the token that marks end of decoding.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
      beam_width:  Python integer, the number of beams.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`.  Optional layer to apply to the RNN output prior
        to storing the result or sampling.
      length_penalty_weight: Float weight to penalize length. Disabled with 0.0.
      coverage_penalty_weight: Float weight to penalize the coverage of source
        sentence. Disabled with 0.0.
      reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the cell
        state will be reordered according to the beam search path. If the
        `TensorArray` can be reordered, the stacked form will be returned.
        Otherwise, the `TensorArray` will be returned as is. Set this flag to
        `False` if the cell state contains `TensorArray`s that are not amenable
        to reordering.
    Raises:
      TypeError: if `cell` is not an instance of `RNNCell`,
        or `output_layer` is not an instance of `tf.layers.Layer`.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)  # pylint: disable=protected-access
    if (output_layer is not None and
        not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._output_layer = output_layer
    self._reorder_tensor_arrays = reorder_tensor_arrays

    if callable(embedding):
      self._embedding_fn = embedding
    else:
      self._embedding_fn = (
          lambda ids: embedding_ops.embedding_lookup(embedding, ids))

    self._end_token = ops.convert_to_tensor(
        end_token, dtype=dtypes.int32, name="end_token")
    if self._end_token.get_shape().ndims != 0:
      raise ValueError("end_token must be a scalar")

    self._batch_size = array_ops.size(start_tokens)
    self._beam_width = beam_width
    self._length_penalty_weight = length_penalty_weight
    self._coverage_penalty_weight = coverage_penalty_weight
    self._initial_cell_state = nest.map_structure(
        self._maybe_split_batch_beams, initial_state, self._cell.state_size)
    self._start_inputs = start_emb_inputs

    self._finished = array_ops.one_hot(
        array_ops.zeros([self._batch_size], dtype=dtypes.int32),
        depth=self._beam_width,
        on_value=False,
        off_value=True,
        dtype=dtypes.bool)

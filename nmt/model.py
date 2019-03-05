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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
from functools import reduce
import numpy as np
from operator import mul

from IPython import embed
import tensorflow as tf

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from .utils import vocab_utils
from .utils import style_utils
from .utils import inference_helpers

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class TrainOutputTuple(collections.namedtuple(
    "TrainOutputTuple", ("train_summary", "train_loss", "predict_count",
                         "global_step", "word_count", "batch_size", "grad_norm",
                         "learning_rate"))):
  """To allow for flexibily in returing different outputs."""
  pass


class EvalOutputTuple(collections.namedtuple(
    "EvalOutputTuple", ("eval_loss", "predict_count", "batch_size"))):
  """To allow for flexibily in returing different outputs."""
  pass


class InferOutputTuple(collections.namedtuple(
    "InferOutputTuple", ("infer_logits", "infer_summary", "sample_id",
                         "sample_words", "sample_style"))):
  """To allow for flexibily in returing different outputs."""
  pass


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               vocab_table,
               reverse_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_vocab_table: Lookup table mapping ids to words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    # Set params
    self._set_params_initializer(hparams, mode, iterator, vocab_table, 
                                 scope, extra_args)

    # Not used in general seq2seq models; when True, ignore decoder & training
    self.extract_encoder_layers = (hasattr(hparams, "extract_encoder_layers")
                                   and hparams.extract_encoder_layers)

    # Train graph
    res = self.build_graph(hparams, scope=scope)
    if not self.extract_encoder_layers:
      self._set_train_or_infer(res, reverse_vocab_table, hparams)

    # Saver
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

  def _set_params_initializer(self,
                              hparams,
                              mode,
                              iterator,
                              vocab_table,
                              scope,
                              extra_args=None):
    """Set various params for self and initialize."""
    assert isinstance(iterator, iterator_utils.StyleBatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.vocab_table = vocab_table

    self.vocab_size = hparams.vocab_size
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major

    if hparams.use_char_encode:
      assert (not self.time_major), ("Can't use time major for"
                                     " char-level inputs.")

    self.dtype = tf.float32
    self.num_sampled_softmax = hparams.num_sampled_softmax

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Set num units
    self.num_units = hparams.num_units

    # Set num styles
    self.num_styles = hparams.num_styles

    # Set num layers
    self.num_encoder_layers = hparams.num_encoder_layers
    self.num_decoder_layers = hparams.num_decoder_layers
    assert self.num_encoder_layers
    assert self.num_decoder_layers

    # Set maxpool window width
    self.maxpool_width = hparams.maxpool_width

    # Set num residual layers
    if hasattr(hparams, "num_residual_layers"):  # compatible common_test_utils
      self.num_encoder_residual_layers = hparams.num_residual_layers
      self.num_decoder_residual_layers = hparams.num_residual_layers
    else:
      self.num_encoder_residual_layers = hparams.num_encoder_residual_layers
      self.num_decoder_residual_layers = hparams.num_decoder_residual_layers

    # Batch size
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Global step
    self.global_step = tf.Variable(0, trainable=False)

    # Loss weighting
    self.lambda_ae = tf.Variable(hparams.lambda_ae, trainable=False)
    self.lambda_bt = tf.Variable(hparams.lambda_bt, trainable=False)

    # Initializer
    self.random_seed = hparams.random_seed
    initializer = model_helper.get_initializer(
        hparams.init_op, self.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    if extra_args and extra_args.encoder_emb_lookup_fn:
      assert False
      self.encoder_emb_lookup_fn = extra_args.encoder_emb_lookup_fn
    else:
      self.encoder_emb_lookup_fn = tf.nn.embedding_lookup
    self.init_embeddings(hparams, scope)

    # Style Table
    self.style_metadata = hparams.style_metadata
    self.style_file = hparams.style_file
    self.style_table = style_utils.create_style_table(self.style_file)

    # Style Embeddings
    self.style_embedding = style_utils.create_style_embedding(
        num_styles=self.num_styles, embed_size=self.num_units)

    # Initialize variables for sampling
    self._init_style_sampling_variables()

  def _set_train_or_infer(self, res, reverse_vocab_table, hparams):
    """Set up training and inference."""
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_context_state, self.sample_id = res
      self.sample_words = reverse_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrange for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # beta1 for Adam
      self.beta1 = tf.constant(hparams.beta1)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
      else:
        raise ValueError("Unknown optimizer type %s" % hparams.optimizer)

      # Gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm_summary = grad_norm_summary
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

      # Summary
      self.train_summary = self._get_train_summary()
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    utils.print_out("Format: <name>, <shape>, <(soft) device placement>")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_decay_info(self, hparams):
    """Return decay info based on decay_scheme."""
    if hparams.decay_scheme in ["luong5", "luong10", "luong234"]:
      decay_factor = 0.5
      if hparams.decay_scheme == "luong5":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 5
      elif hparams.decay_scheme == "luong10":
        start_decay_step = int(hparams.num_train_steps / 2)
        decay_times = 10
      elif hparams.decay_scheme == "luong234":
        start_decay_step = int(hparams.num_train_steps * 2 / 3)
        decay_times = 4
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / decay_times)
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    return start_decay_step, decay_steps, decay_factor

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):

    # NOTES: 
    #   - Add command line arg for style specific word embeddings

    """Init embeddings."""
    # Old embedding -- just to keep the peace for now
    self.embedding = model_helper.create_emb_for_encoder_and_decoder(
      vocab_size=self.vocab_size,
      embed_size=self.num_units,
      num_enc_partitions=hparams.num_enc_emb_partitions,
      num_dec_partitions=hparams.num_dec_emb_partitions,
      vocab_file=hparams.vocab_file,
      embed_file=hparams.embed_file,
      use_char_encode=hparams.use_char_encode,
      scope=scope,
      name="embedding")

    # New style specific embeddings
    style_specific_embeddings = []
    for i in range(self.num_styles):
      style_specific_embeddings.append(
        tf.expand_dims(
          model_helper.create_emb_for_encoder_and_decoder(
            vocab_size=self.vocab_size,
            embed_size=self.num_units,
            num_enc_partitions=hparams.num_enc_emb_partitions,
            num_dec_partitions=hparams.num_dec_emb_partitions,
            vocab_file=hparams.vocab_file,
            embed_file=hparams.embed_file,
            use_char_encode=hparams.use_char_encode,
            scope=scope,
            name="embedding"+str(i)), axis=1))
    self.embedding_tensor = tf.concat(style_specific_embeddings, axis=1)

  def _init_style_sampling_variables(self):
    ## Get num of the possible styles for each attribute
    self.modulo_constants = [] # list of all the styles as strings
    for attribute_dict in self.style_metadata['attributes']:
      self.modulo_constants.append(len(attribute_dict[
                                            list(attribute_dict.keys())[0]]))

    # total number of unique styles
    self.num_attributes = len(self.modulo_constants)
    self.num_uniq_styles = reduce(mul, self.modulo_constants, 1)

    # start indices for each attribute
    self.start_indices = [sum(self.modulo_constants[:i]) \
                                      for i in range(self.num_attributes)]

    # integer divisors for conversion to integer representation (as tensor)
    self.integer_divisors = [reduce(mul, self.modulo_constants[:i], 1) \
                                      for i in range(self.num_attributes)]


  def _get_train_summary(self):
    """Get train summary."""
    train_summary = tf.summary.merge(
        [tf.summary.scalar("lr", self.learning_rate),
         tf.summary.scalar("train_loss", self.train_loss)] +
        self.grad_norm_summary)
    return train_summary

  def train(self, sess):
    """Execute train graph."""
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    output_tuple = TrainOutputTuple(train_summary=self.train_summary,
                                    train_loss=self.train_loss,
                                    predict_count=self.predict_count,
                                    global_step=self.global_step,
                                    word_count=self.word_count,
                                    batch_size=self.batch_size,
                                    grad_norm=self.grad_norm,
                                    learning_rate=self.learning_rate)
    return sess.run([self.update, output_tuple])

  def eval(self, sess):
    """Execute eval graph."""
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    output_tuple = EvalOutputTuple(eval_loss=self.eval_loss,
                                   predict_count=self.predict_count,
                                   batch_size=self.batch_size)
    return sess.run(output_tuple)

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss_tuple, final_context_state, sample_id),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: loss = the total loss / batch_size.
        final_context_state: the final state of decoder RNN.
        sample_id: sampling indices.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# Creating %s graph ..." % self.mode)

    # Projection
    if not self.extract_encoder_layers:
      with tf.variable_scope(scope or "build_network"):
        with tf.variable_scope("decoder/output_projection"):
          self.output_layer = tf.layers.Dense(
              self.vocab_size, use_bias=False, name="output_projection")

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=self.dtype,
        reuse=tf.AUTO_REUSE):

      assert not self.extract_encoder_layers

      sequence = self.iterator.source
      sequence_length = self.iterator.source_sequence_length
      style_labels = self.iterator.style_labels

      self.inp_emb = self._lookup_embedding(sequence, style_labels)



      self.target_style_labels = self._sample_style_labels(style_labels)

      if hparams.language_model:
        with tf.variable_scope("generator", dtype=self.dtype,
            reuse=tf.AUTO_REUSE):
          encoder_outputs = None
          encoder_state = None

          (logits, decoder_cell_outputs, 
           sample_id, final_context_state) = (self._build_decoder(hparams,
             encoder_outputs, encoder_state, sequence_length, style_labels))

      # Train or eval
      elif self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.variable_scope("generator", dtype=self.dtype,
            reuse=tf.AUTO_REUSE):
          ## Auto-encode 
          # Noise input sequence
          noisy_sequence = self._noise_sequence(sequence,
              word_drop=hparams.word_drop,
              permutation_limit=hparams.permutation_limit)

          # Encode input sequence
          encoder_outputs, encoder_state = self._build_encoder(hparams,
              noisy_sequence, sequence_length, style_labels)

          # Apply temporal max-pooling
          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
            
          encoder_outputs = tf.layers.max_pooling1d(encoder_outputs,
              self.maxpool_width, self.maxpool_width, name="maxpool")

          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])

          # Decode input sequence
          (ae_logits, ae_decoder_cell_outputs, 
           ae_sample_id, ae_final_context_state) = (self._build_decoder(hparams,
             encoder_outputs, encoder_state, sequence_length, style_labels))

          ## Back-translate
          encoder_outputs, encoder_state = self._build_encoder(hparams,
              sequence, sequence_length, style_labels)

          # Apply temporal max-pooling
          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
            
          encoder_outputs = tf.layers.max_pooling1d(encoder_outputs,
              self.maxpool_width, self.maxpool_width, name="maxpool")

          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])

          # Decode input sequence
          sample_logits, _, sample_id, self.final_context_state = \
              self._build_decoder(hparams, encoder_outputs,
              encoder_state, sequence_length, self.target_style_labels,
              back_trans=True)

          if self.time_major:
            sample_logits = tf.transpose(sample_logits, [1, 0, 2])

          # Soft sampled seq for feature extractor
          soft_sampled_seq = self._lookup_soft_embedding(sample_logits,
                                                     self.target_style_labels)

          # Hard sampled seq for back-translation
          sampled_pseudo_sequence = sample_id

          # transpose it back to batch major if time major.
          # encoder will automatically transpose it back if time_major
          if self.time_major:
            sampled_pseudo_sequence = tf.transpose(sampled_pseudo_sequence)

          ## Stop gradient from back-propagating all the way through
          tf.stop_gradient(sampled_pseudo_sequence)

          ## Get sampled pseudo sequence's length
          sampled_seq_length = tf.fill([tf.shape(sampled_pseudo_sequence)[0]],
                                        tf.shape(sampled_pseudo_sequence)[1])

          self.sampled_pseudo_sequence = sampled_pseudo_sequence

          encoder_outputs, encoder_state = self._build_encoder(hparams,
              sampled_pseudo_sequence, sampled_seq_length,
              self.target_style_labels)

          # Apply temporal max-pooling
          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
            
          encoder_outputs = tf.layers.max_pooling1d(encoder_outputs,
              self.maxpool_width, self.maxpool_width, name="maxpool")

          if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])

          # Decode input sequence
          (bt_logits, bt_decoder_cell_outputs, 
           bt_sample_id, bt_final_context_state) = (self._build_decoder(hparams,
             encoder_outputs, encoder_state, sampled_seq_length, style_labels))

        # Discriminator
        with tf.variable_scope("feature_extractor", dtype=self.dtype,
            reuse=tf.AUTO_REUSE):
          self._build_feature_extractor(hparams)
          real_sent_emb = self._lookup_embedding(sequence, style_labels)
          real_sent_reps = self._extract_sent_feats(real_sent_emb)

          fake_sent_reps = self._extract_sent_feats(soft_sampled_seq)

          self.rsr = real_sent_reps
          self.fsr = fake_sent_reps

          # Loss: IPOT(real_sent_reps, fake_sent_reps)


      # Inference
      else:
        with tf.variable_scope("generator", dtype=self.dtype,
            reuse=tf.AUTO_REUSE):
          encoder_outputs, encoder_state = self._build_encoder(hparams,
              sequence, sequence_length, style_labels)

          (logits, decoder_cell_outputs, 
            sample_id, final_context_state) = self._build_decoder(hparams,
                encoder_outputs, encoder_state, sequence_length,
                self.target_style_labels, back_trans=False)


      ## Loss
      if (hparams.language_model          
          and self.mode != tf.contrib.learn.ModeKeys.INFER):
        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                   self.num_gpus)):
          loss = self._compute_loss(logits, decoder_cell_outputs)

      elif self.mode != tf.contrib.learn.ModeKeys.INFER:
        logits = ae_logits
        sample_id = ae_sample_id
        final_context_state = ae_final_context_state
        #(logits, sample_id, final_context_state) = None, None, None

        with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                   self.num_gpus)):
          ae_loss = self._compute_loss(ae_logits, ae_decoder_cell_outputs)
          bt_loss = self._compute_loss(bt_logits, bt_decoder_cell_outputs)
  
          
          lambda_ae = (1 - ((self.lambda_ae 
                                 * tf.cast(self.global_step, tf.float32))
                                / hparams.num_train_steps))

          loss = (lambda_ae * ae_loss) + (self.lambda_bt * bt_loss)
      else:
        loss = tf.constant(0.0)

      return logits, loss, final_context_state, sample_id

  def _build_language_model(self, hparams):
    ## TODO: make this work for style -- this is currently wrong
    raise NotImplementedError("Language model for style not implemented.")

    ## Decoder
    logits, decoder_cell_outputs, sample_id, final_context_state = (
      self._build_decoder(None, None, hparams))

    ## Loss
    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                 self.num_gpus)):
        loss = self._compute_loss(logits, decoder_cell_outputs)
    else:
      loss = tf.constant(0.0)

    return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams, sequence, sequence_length, style_labels):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _init_style_state(self, hparams, switch_style=False, prev_state=None):
    assert hparams.unit_type == "lstm" \
           or hparams.unit_type == "layer_norm_lstm"

    return None

  def _noise_sequence(self, sequence, word_drop=0.1, permutation_limit=3):
    """ Noise model from Lample et al. 2017. """
    # drop words from sequence
    word_drop_mask = tf.random.uniform(tf.shape(sequence)) > word_drop
    word_drop_tensor = tf.fill(tf.shape(sequence), vocab_utils.UNK_ID)
    noisy_sequence = tf.where(word_drop_mask, sequence, word_drop_tensor)

    # shuffle slightly
    q = tf.range(tf.cast(tf.shape(sequence)[1], tf.float32))
    q = tf.tile(q, tf.fill([1], tf.shape(sequence)[0]))
    q = tf.reshape(q, tf.shape(sequence))
    q = q + (permutation_limit + 1) * tf.random.uniform(tf.shape(sequence))
    sigma = tf.contrib.framework.argsort(q)

    batch_index = tf.range(tf.shape(sequence)[0])
    batch_index = tf.tile(batch_index, tf.fill([1], tf.shape(sequence)[1]))
    batch_index = tf.transpose(tf.reshape(batch_index,
                               [tf.shape(sequence)[1], tf.shape(sequence)[0]]))

    indices = tf.stack([batch_index, sigma], axis=2)

    noisy_sequence = tf.gather_nd(noisy_sequence, indices)

    return noisy_sequence

  def _sample_style_labels(self, current_style_labels):
    # convert modulo_constants, start_indices, and integer_divisors to tensors
    modulo_constants = tf.broadcast_to(tf.constant(self.modulo_constants),
                                           tf.shape(current_style_labels))
    start_indices = tf.broadcast_to(tf.constant(self.start_indices),
                                       tf.shape(current_style_labels))
    integer_divisors = tf.broadcast_to(tf.constant(self.integer_divisors),
                                       tf.shape(current_style_labels))

    ## STEP 1: Convert all sets of style labels to integer representation
    #inv_rep = lambda r : sum([r*d for r, d in zip(r, integer_divisors)])
    _current_style_labels = current_style_labels - start_indices
    int_style_rep = tf.reduce_sum(tf.multiply(_current_style_labels,
                                              self.integer_divisors), axis=1)

    ## STEP 2: Generate a random number in [1...(num_uniq_styles-1)] for each
    ##          example in batch
    rand_ints = tf.random.uniform(tf.shape(int_style_rep), minval=1,
                                maxval=self.num_uniq_styles-1, dtype=tf.int32)

    ## STEP 3: (STEP 1 + STEP 2) % num_uniq_styles
    new_int_reps = tf.expand_dims(tf.floormod(tf.add(int_style_rep, rand_ints),
                                  tf.constant(self.num_uniq_styles)), axis=1)


    ## STEP 4: Convert STEP 3 batch to style label represenation
    #rep = lambda x : [(x // d) % m for d, m in zip(integer_divisors, modulo_constants)]
    sampled_style_labels = tf.floormod(tf.floordiv(new_int_reps,
                                          integer_divisors), modulo_constants)

    sampled_style_labels = sampled_style_labels + start_indices

    return sampled_style_labels

  def _lookup_embedding(self, sequence, style_labels):
    ### NOTE: this function assumes sequence is fed in batch_major order

    one_hot_rep = tf.reduce_sum(tf.one_hot(style_labels,
                                                self.num_styles), axis=1)

    seq_emb = tf.nn.embedding_lookup(self.embedding_tensor, sequence)
    seq_emb = tf.transpose(seq_emb, [0, 2, 1, 3])
    seq_emb = tf.boolean_mask(seq_emb, one_hot_rep)
    seq_emb = tf.reshape(seq_emb, [tf.shape(style_labels)[0],
                                   tf.shape(style_labels)[1],
                                   -1, self.num_units])
    seq_emb = tf.reduce_mean(seq_emb, axis=1)

    ## NOTE: If only one style we do this
    #seq_emb = tf.nn.embedding_lookup(self.embedding, sequence)

    return seq_emb

  def _lookup_dynamic_decode_embedding(self, sequence, style_labels):
    one_hot_rep = tf.reduce_sum(tf.one_hot(style_labels,
                                                self.num_styles), axis=1)

    seq_emb = tf.nn.embedding_lookup(self.embedding_tensor, sequence)
    seq_emb = tf.boolean_mask(seq_emb, one_hot_rep)
    seq_emb = tf.reshape(seq_emb, [tf.shape(style_labels)[0],
                                   tf.shape(style_labels)[1],
                                   self.num_units])
    seq_emb = tf.reduce_mean(seq_emb, axis=1)

    return seq_emb

  def _lookup_soft_embedding(self, logits, style_labels):
    ### NOTE: this function assumes sequence is fed in batch_major order
    one_hot_rep = tf.reduce_sum(tf.one_hot(style_labels,
                                           self.num_styles), axis=1)

    soft_emb = tf.tensordot(logits, self.embedding_tensor, axes=[[2], [0]])

    soft_emb = tf.transpose(soft_emb, [0, 2, 1, 3])
    soft_emb = tf.boolean_mask(soft_emb, one_hot_rep)
    soft_emb = tf.reshape(soft_emb, [tf.shape(style_labels)[0],
                                     tf.shape(style_labels)[1],
                                     -1, self.num_units])
    soft_emb = tf.reduce_mean(soft_emb, axis=1)

    ## NOTE: If only one style we do this
    ## << What do I do in this case? >> 

    return soft_emb

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.max_len_infer:
      maximum_iterations = hparams.max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, hparams, encoder_outputs, init_state,
                     sequence_length, style_labels, back_trans=False):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      hparams: The Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      init_state: The initial state of the encoder.
      sequence_length: The length of the sequence fed to encoder.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)
    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, init_state,
          sequence_length)

      # Optional ops depends on which mode we are in and which loss function we
      # are using.
      logits = tf.no_op()
      decoder_cell_outputs = None

      ## NOTE: `back_trans=True` means we're performing inference for the
      ##         purposes of back-translation.

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER and not back_trans:
        target_input = iterator.target_input
        time_axis = 1

        if self.time_major:
          # decoder_emp_inp: [max_time, batch_size, num_units]
          target_input = tf.transpose(target_input)
          time_axis = 0

        decoder_emb_inp = self._lookup_embedding(target_input, style_labels)

        # Pass style embedding as <SOS> to decoder
        style_emb_inp = tf.reduce_mean(tf.nn.embedding_lookup(
            self.style_embedding, style_labels), axis=1)
        style_emb_inp = tf.expand_dims(style_emb_inp, axis=time_axis)
        if self.time_major:
          decoder_emb_inp_mod = tf.concat([style_emb_inp,
                                           decoder_emb_inp[1:,:,:]],
                                          axis=time_axis)
        else:
          decoder_emb_inp_mod = tf.concat([style_emb_inp,
                                           decoder_emb_inp[:,1:,:]],
                                          axis=time_axis)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp_mod, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        if self.num_sampled_softmax > 0:
          # Note: this is required when using sampled_softmax_loss.
          decoder_cell_outputs = outputs.rnn_output

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        num_layers = self.num_decoder_layers
        num_gpus = self.num_gpus
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        # Colocate output layer with the last RNN cell if there is no extra GPU
        # available. Otherwise, put last layer on a separate GPU.
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_layer(outputs.rnn_output)

        if self.num_sampled_softmax > 0:
          logits = tf.no_op()  # unused when using sampled softmax loss.

      ## Inference
      else:

        ### NOTE: Somehow have to get infer_mode to work with global_step and
        ###        back_trans

        #infer_mode = "sample" if back_trans else hparams.infer_mode
        infer_mode = "sample"

        self.infer_mode = infer_mode
        
        style_emb_inp = tf.reduce_mean(tf.nn.embedding_lookup(
            self.style_embedding, style_labels), axis=1)

        start_tokens=tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        self.style_emb_inp = style_emb_inp

        utils.print_out(
            "  decoder: infer_mode=%sbeam_width=%d, length_penalty=%f" % (
                infer_mode, hparams.beam_width, hparams.length_penalty_weight))

        embedding_fn = lambda ids: self._lookup_dynamic_decode_embedding(ids, style_labels)

        if infer_mode == "beam_search":
          beam_width = hparams.beam_width
          length_penalty_weight = hparams.length_penalty_weight

          assert False
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
        elif infer_mode == "sample":
          # Helper
          sampling_temperature = tf.constant(hparams.sampling_temperature)
          sampling_temperature = (tf.cast(self.global_step, tf.float32)
                                  * sampling_temperature
                                  / hparams.num_train_steps)
          sampling_temperature = tf.cast(sampling_temperature, tf.float32)

          helper = inference_helpers.SampleInferenceHelper(
              embedding_fn, style_emb_inp, end_token,
              softmax_temperature=sampling_temperature,
              seed=self.random_seed)
        elif infer_mode == "greedy":
          helper = inference_helpers.GreedyInferenceHelper(
              embedding_fn, style_emb_inp, end_token)
        else:
          raise ValueError("Unknown infer_mode '%s'", infer_mode)

        if infer_mode != "beam_search":
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if infer_mode == "beam_search":
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, decoder_cell_outputs, sample_id, final_context_state

  def _build_feature_extractor(self, hparams):
    num_filters = 128
    window_sizes = [2, 3, 5, 7]
    self.conv_layers = []

    for window_size in window_sizes:
      self.conv_layers.append(tf.keras.layers.Conv2D(num_filters,
                                                [window_size, self.num_units],
                                                data_format="channels_last",
                                                activation="tanh",
                                                name=str(window_size)))
    return

  def _extract_sent_feats(self, embed_input):
    embed_input = tf.expand_dims(embed_input, axis=3)

    conv_outputs = []
    for conv_layer in self.conv_layers:
      x = conv_layer.apply(embed_input)
      x = tf.reduce_max(x, axis=1)
      x = tf.squeeze(x)
      conv_outputs.append(x)

    return tf.concat(conv_outputs, axis=1)

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, init_state,
                          source_sequence_length, base_gpu=0):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      init_state: The initial state of the encoder.
      sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder and the intial state of
      the decoder RNN.
    """
    pass

  def _softmax_cross_entropy_loss(
      self, logits, decoder_cell_outputs, labels):
    """Compute softmax loss or sampled softmax loss."""
    if self.num_sampled_softmax > 0:

      is_sequence = (decoder_cell_outputs.shape.ndims == 3)

      if is_sequence:
        labels = tf.reshape(labels, [-1, 1])
        inputs = tf.reshape(decoder_cell_outputs, [-1, self.num_units])

      crossent = tf.nn.sampled_softmax_loss(
          weights=tf.transpose(self.output_layer.kernel),
          biases=self.output_layer.bias or tf.zeros([self.vocab_size]),
          labels=labels,
          inputs=inputs,
          num_sampled=self.num_sampled_softmax,
          num_classes=self.vocab_size,
          partition_strategy="div",
          seed=self.random_seed)

      if is_sequence:
        if self.time_major:
          crossent = tf.reshape(crossent, [-1, self.batch_size])
        else:
          crossent = tf.reshape(crossent, [self.batch_size, -1])

    else:
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)

    return crossent

  def _compute_loss(self, logits, decoder_cell_outputs):
    """Compute optimization loss."""
    target_output = self.iterator.target_output
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)

    crossent = self._softmax_cross_entropy_loss(
        logits, decoder_cell_outputs, target_output)

    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=self.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)
    return loss

  def _get_infer_summary(self, hparams):
    del hparams
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    output_tuple = InferOutputTuple(infer_logits=self.infer_logits,
                                    infer_summary=self.infer_summary,
                                    sample_id=self.sample_id,
                                    sample_words=self.sample_words,
                                    sample_style=self.target_style_labels)
    return sess.run(output_tuple)

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    output_tuple = self.infer(sess)
    sample_words = output_tuple.sample_words
    infer_summary = output_tuple.infer_summary
    sample_style = output_tuple.sample_style

    # make sure outputs is of shape [batch_size, time] or [beam_width,
    # batch_size, time] when using beam search.
    if self.time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:
      # beam search output in [batch_size, time, beam_width] shape.
      sample_words = sample_words.transpose([2, 0, 1])
    return sample_words, infer_summary, sample_style

  def build_encoder_states(self, include_embeddings=False):
    """Stack encoder states and return tensor [batch, length, layer, size]."""
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    if include_embeddings:
      stack_state_list = tf.stack(
          [self.encoder_emb_inp] + self.encoder_state_list, 2)
    else:
      stack_state_list = tf.stack(self.encoder_state_list, 2)

    # transform from [length, batch, ...] -> [batch, length, ...]
    if self.time_major:
      stack_state_list = tf.transpose(stack_state_list, [1, 0, 2, 3])

    return stack_state_list


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """
  def _build_encoder(self, hparams, sequence, sequence_length, style_labels):
    """Build an encoder from a sequence.

    Args:
      hparams: hyperparameters.
      initial_state: initial state for RNN - contains style embedding.
      sequence: tensor with input sequence data.
      sequence_length: tensor with length of the input sequence.

    Returns:
      encoder_outputs: RNN encoder outputs.
      encoder_state: RNN encoder state.

    Raises:
      ValueError: if encoder_type is neither "uni" nor "bi".
    """
    utils.print_out("# Build a basic encoder")

    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      
      # Look up embeddings for each token in sequence
      self.encoder_emb_inp = self._lookup_embedding(sequence, style_labels)

      if self.time_major:
        sequence = tf.transpose(sequence)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(hparams, num_layers,
                                        num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            self.encoder_emb_inp,
            initial_state=None,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=self.encoder_emb_inp,
                sequence_length=sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          #$# alternatively concat forward and backward states
          #encoder_state = []
          #for layer_id in range(num_bi_layers):
          #  encoder_state.append(bi_encoder_state[0][layer_id])  # forward
          #  encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          #encoder_state = tuple(encoder_state)
          
          ## FOR OUR USE CASE: Just want the top layer
          encoder_state = []
          layer_id = num_bi_layers - 1
          encoder_state.append(bi_encoder_state[0][layer_id])  # forward
          encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)

      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    # Use the top layer for now
    self.encoder_state_list = [encoder_outputs]

    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, init_state,
                          source_sequence_length, base_gpu=0):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=self.num_decoder_layers,
        num_residual_layers=self.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn,
        base_gpu=base_gpu)

    if hparams.language_model:
      encoder_state = cell.zero_state(self.batch_size, self.dtype)
    elif not hparams.pass_hidden_state:
      raise ValueError("For non-attentional model, "
                       "pass_hidden_state needs to be set to True")

    # For beam search, we need to replicate encoder infos beam_width times
    if (self.mode == tf.contrib.learn.ModeKeys.INFER and
        hparams.infer_mode == "beam_search"):
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state

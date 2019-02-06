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

from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import lookup_ops

from IPython import embed


def create_style_table(style_file):
  """
  ### To query table:
  sample_attr = tf.constant(['positive', 'male', 'asian'])
  ids = style_table.lookup(sample_attr)
  """
  return lookup_ops.index_table_from_file(style_file)


def create_style_embedding(num_styles, embed_size):
  ## Create embedding
  style_embedding = tf.get_variable("style_embeddings",
      [num_styles, embed_size])

  """
  ### To query style embeddings
  embedded_style_ids = tf.nn.embedding_lookup(style_embedding, style_ids)
  """
  return style_embedding
  

"""

Functions and components that can be slotted into tensorflow models.

TODO: Write functions for various types of attention. Stacked-BiLSTM, CNN, GRU, Reuse-Block
TODO: Embedding-Block
"""

import tensorflow as tf
from tensorflow import variable_scope

from model.cells import DecoderGRUCell


def _length(sequence):
  """
  Get true _length of sequences (without padding), and mask for true-_length in max-_length.

  Input of shape: (batch_size, max_seq_length, hidden_dim)
  Output shapes,
  _length: (batch_size)
  mask: (batch_size, max_seq_length, 1)
  """
  populated = tf.sign(tf.abs(sequence))
  length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
  mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
  return length, mask

def _get_attention_weights(hidden_states, head, seq_length, mask, name):
  with tf.name_scope(name):
    scores_i = []
    for j in range(seq_length):
      score_ij = tf.reduce_sum(tf.multiply(hidden_states[j], head), 1, keep_dims=True)
      scores_i.append(score_ij)
    scores_i = tf.stack(scores_i, axis=1)
    alpha_i = masked_softmax(scores_i, mask)
  return alpha_i

def multiHeadSelfAttention(hidden_states, seq_length, max_seq_len, mask, dim, n_heads, name):
  aspects = []

  W = tf.Variable(tf.random_normal([dim,dim]), name = "%s_W" % name)
  b = tf.Variable(tf.random_normal([dim]), name = "%s_b" % name)

  h = tf.unstack(hidden_states, axis=1)
  h_bar = []
  for i in range(max_seq_len):
    h_bar_i = tf.matmul(h[i], W) + b
    h_bar.append(h_bar_i)

  for i in range(n_heads):
    head = tf.Variable(tf.random_normal([dim], stddev=0.1),
                       name = "%s_aspect_%s" % (name, str(i)))
    alpha_i = _get_attention_weights(h_bar, head, max_seq_len, mask, name)
    aspect = tf.reduce_sum(tf.multiply(alpha_i, hidden_states),axis=1) # attention weighted repr. w.r.t. one aspect
    aspects.append(aspect)
  return aspects, [h]


def RNN(inputs, dim, seq_len, name, rnn_cell):
  with tf.name_scope(name):
    rnn_cell = rnn_cell(num_units=dim)
    hidden_states, cell_states = tf.nn.dynamic_rnn(rnn_cell, inputs=inputs, sequence_length=seq_len, dtype=tf.float32,
                                                   scope=name)
  return hidden_states, cell_states

def LSTM(inputs, dim, seq_len, name="LSTM"):
  """
  An LSTM layer. Returns hidden states and cell states as a tuple.

  Ouput shape of hidden states: (batch_size, max_seq_length, hidden_dim)
  Same shape for cell states.
  """
  return RNN(inputs, dim, seq_len, name, tf.contrib.rnn.LSTMCell)

def GRU(inputs, dim, seq_length, name="GRU"):
  return RNN(inputs, dim, seq_length, name, tf.contrib.rnn.GRUCell)


def biGRU(inputs, dim, seq_len, name="biGRU"):
  cell = tf.contrib.rnn.GRUCell # does tanh by default
  return biRNN(inputs, cell, dim, seq_len, name)

def biLSTM(inputs, dim, seq_len, name="biLSTM"):
  cell = tf.contrib.rnn.LSTMCell
  return biRNN(inputs, cell, dim, seq_len, name)

def biRNN(inputs, cell, dim, seq_len, name="biRNN"):
  """
  A Bi-Directional RNN layer. Returns forward and backward hidden states as a tuple, and cell states as a tuple.

  Ouput of hidden states: [(batch_size, max_seq_length, hidden_dim), (batch_size, max_seq_length, hidden_dim)]
  Same shape for cell states.
  """
  rnn_fwd = cell(num_units=dim)
  rnn_bwd = cell(num_units=dim)
  hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fwd, cell_bw=rnn_bwd, inputs=inputs,
                                                               sequence_length=seq_len, dtype=tf.float32, scope=name)
  return hidden_states, cell_states


def last_output(output, true_length, name=""):
  """
  To get the last hidden layer form a dynamically unrolled RNN.
  Input of shape (batch_size, max_seq_length, hidden_dim).

  true_length: Tensor of shape (batch_size). Such a tensor is given by the _length() function.
  Output of shape (batch_size, hidden_dim).
  """
  var_scope = "last_output" if not name else "last_output_" + name
  with tf.variable_scope(var_scope):
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length, on_value=1., off_value=0.), -1)
    return tf.reduce_sum(tf.multiply(output, length_mask), 1)

def masked_softmax(scores, mask, name=""):
  """
  Used to calculcate a softmax score with true sequence _length (without padding), rather than max-sequence _length.

  Input shape: (batch_size, max_seq_length, hidden_dim).
  mask parameter: Tensor of shape (batch_size, max_seq_length). Such a mask is given by the _length() function.
  """
  global asd
  var_scope = "masked_softmax" if not name else "masked_softmax_" + name
  with tf.variable_scope(var_scope):
    asd = tf.subtract(scores, tf.reduce_max(scores, 1, keep_dims=True), name="subtract")
    numerator = tf.exp(asd) * mask
    denominator = tf.reduce_sum(numerator, 1, keep_dims=True)
    weights = tf.div(numerator, denominator)
    return weights

def GRU_Decoder(inputs, dim, seq_len, name="GRU_Decoder"):
  cell = DecoderGRUCell
  return RNN_Decoder(inputs, dim, seq_len, name)

def RNN_Decoder(inputs, dim, seq_len, cell, name="RNN_Decoder"):
  with tf.name_scope(name):
    rnn_decoder_cell = cell(num_units=dim)
    helper = tf.contrib.seq2seq.TrainingHelper(inputs, seq_len, time_major=False)

    # decoder = tf.contrib.seq2seq.BasicDecoder(rnn_decoder_cell, helper, output)

def tf_cosine(inputs_a, inputs_b, name=""):
  """
  Assumes batches of sentence representations in shape (batch, sent_emb_size)
  :param inputs_a: batches of first input representation, e.g. result from shared encoder
  :param inputs_b: batches of second input representation, e.g. result from private encoder
  :param name: variable scope
  :return: cosine similarities
  """
  var_scope = "tf_cosine" if not name else "tf_cosine_" + name
  with tf.variable_scope(var_scope):
    unit_inputs_a = tf.nn.l2_normalize(inputs_a, axis=1)
    unit_inputs_b = tf.nn.l2_normalize(inputs_b, axis=1)
    cosines = tf.reduce_sum(tf.multiply(unit_inputs_a, unit_inputs_b), axis=1)
    return cosines


def sentence2onehot(vocabindices_seqlen, vocab_size, return_dense=False, name=""):
  """
  Takes an array of vocabulary indices and turns it into a one-hot sparse representation. E.g. for a sentence
  represented by its vocabulary token ids [0, 1, 2] is turned into a sentence matrix [[0,0,1],[0,1,0],[0,0,1]].
  Use tf.map_fn to apply this to a batch of sentences in order to obtain a tfidf tensor.
  :param vocabindices_seqlen: tuple of a list of word indices in vocabulary and actual sequence length before padding
  :param vocab_size: determines the length of tfidf vectors, i.e. the max value that can appear in vocabulary_indices
  :param return_dense: whether the output matrix should be a sparse or dense matrix
  :param name: variable scope suffix
  :return:
  """
  vocabulary_indices = vocabindices_seqlen[0]
  seq_len = tf.cast(vocabindices_seqlen[1], dtype=tf.int32)
  var_scope = "sentence2onehot" if not name else "sentence2onehot_" + name
  with tf.variable_scope(var_scope):
    max_seq_len = tf.shape(vocabulary_indices)[0]
    seq_len64 = tf.cast(max_seq_len, dtype=tf.int64)
    vocab_size64 = tf.cast(vocab_size, dtype=tf.int64)
    vocab_indices_expanded = tf.expand_dims(vocabulary_indices, -1)

    # ones = tf.reshape(tf.ones_like(vocab_indices_expanded), [-1])  # a.k.a. tfidf values (in future versions)
    ones = tf.ones(seq_len, dtype=tf.int64) # TODO: this needs to become float32 later
    # seq_len32 = tf.cast(seq_len,tf.int32) # TODO: maybe there's some better type checking at an earlier point
    zeros = tf.zeros(max_seq_len - seq_len, dtype=tf.int64)
    values = tf.concat([ones, zeros], 0)

    column_indices = tf.range(tf.shape(vocabulary_indices)[0])
    column_indices_expanded = tf.expand_dims(column_indices, -1)

    tfidf_indices = tf.concat(values=[column_indices_expanded,vocab_indices_expanded], axis=1)
    tfidf_indices_int64 = tf.cast(tfidf_indices, dtype=tf.int64)
    sparse = tf.SparseTensor(indices=tfidf_indices_int64, values=values, dense_shape=[seq_len64, vocab_size64])
    # f([vocabulary_indices, tf.sparse_tensor_to_dense(sparse), seq_len])
    if return_dense:
      return tf.sparse_tensor_to_dense(sparse)
    else:
      return sparse

def hadamard_product(vector_a, vector_b, name=""):
  """
  Given vectors a and b return vector h = [a; b; |a-b|; a*b]
  :param vector_a: batches of vector a
  :param vector_b: batches of vector b
  :param name: variable scope name
  :return: hadamart product of a and b
  """
  var_scope = "hadamard_product" if not name else "hadamard_product_" + name
  with tf.variable_scope(var_scope):
    diff = tf.abs(tf.subtract(vector_a, vector_b))
    mul = tf.multiply(vector_a, vector_b)
    h = tf.concat([vector_a, vector_b, diff, mul], axis=1)
    return h

def mlp(inputs, layer_sizes, activation = tf.nn.tanh, initializer=tf.contrib.layers.xavier_initializer(), name = "mlp"):
  """
  Multi Layer Perceptron, implements a chain of len(layer_sizes) dense layers. The value layer_sizes[-1] determines
  the dimensionality in which the output is projected.
  :param inputs: batches of input vectors
  :param layer_sizes: number of hidden units
  :param activation: non-linearity
  :param initializer: kernel initializer, c.f. arg in tf.layers.dense
  :param name:
  :return:
  """
  layers = []
  with tf.name_scope(name):
    for i in range(len(layer_sizes)):
      if i == 0:
       layers.append(tf.layers.dense(tf.reshape(inputs, [12,-1]), layer_sizes[i], activation, use_bias=True,
                                     kernel_initializer=initializer, name=name + "_layer_" + str(i+1),
                                     reuse=tf.AUTO_REUSE))
      else:
       layers.append(tf.layers.dense(layers[-1], layer_sizes[i], activation, use_bias=True,
                                     kernel_initializer=initializer, name=name + "_layer_" + str(i+1),
                                     reuse=tf.AUTO_REUSE))
    return layers[-1]



# body call of the while_loop for overlapping slicing (tf_overlap_slice)
def _slice_multiple(time, slices, inputs):
  input_shape = inputs.get_shape()
  zeros_after = len(inputs.get_shape()) - (dimsl + 1)
  begin = tf.concat((tf.zeros(shape=(dimsl), dtype = tf.int64), [time], tf.zeros(shape=(zeros_after), dtype = tf.int64)), axis = 0)

  sl = tf.slice(inputs, begin, input_shape[:dimsl].concatenate(slsize).concatenate(input_shape[dimsl+1:]))
  print(sl.get_shape())
  slices=tf.concat([slices, [sl]], axis = 0)
  print(slices.get_shape())

  return (time + stride), slices, inputs

# gets multiple slices of the tensor over one dimension (taking all other dimensions completely).
# the slice_window_size vs. stride_size determines whether (and how much) the slices overlap
def tf_overlap_slice(input, dimension, slice_window_size, dim_size = None, stride_size = 1, name="overlap_slice"):
  with tf.variable_scope("cosines_" + name):
    if dim_size is None:
       dim_size = input.get_shape()[dim_size]

    input_shape = input.get_shape()
    slices_shape_invariant = tf.TensorShape([None]).concatenate(input_shape[:dimension].concatenate(slice_window_size).concatenate(input_shape[dimension+1:]))
    time=tf.constant(0, dtype = tf.int64)

    global dimsl, slsize, stride
    dimsl = dimension
    slsize = slice_window_size
    stride = stride_size

    slices = tf.Variable(tf.zeros(tf.TensorShape([1]).concatenate(slices_shape_invariant[1:]), dtype = tf.float32))

    t, all_slices, inp = tf.while_loop(cond=lambda time, *_: tf.add(time, slice_window_size) <= dim_size,
                                       body=_slice_multiple,
                                       loop_vars=(time, slices, input),
                                       shape_invariants=(time.get_shape(), slices_shape_invariant, input.get_shape()))

    return all_slices[1:]

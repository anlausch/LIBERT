import tensorflow as tf

class BiRNN(tf.layers.Layer):

  def __init__(self, reduce_fn=tf.reduce_max,
               cell=tf.contrib.rnn.GRUCell,
               trainable=True,
               name=None,
               **kwargs):
    super(BiRNN,self).__init__(trainable=trainable,
                               name=name,
                               **kwargs)
    self._reduce_fn =reduce_fn
    self._cell = cell

  def build(self, input_shape=None):
    with tf.variable_scope("test", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False) as scope:
      rnn_fwd = self._cell(num_units=1000)
      rnn_bwd = self._cell(num_units=1000)
      self.fw = rnn_fwd
      # self.bw = rnn_bwd
      self.built = True

  def call(self, inputs, lengths, reuse=False):
    # with  tf.variable_scope("asd", reuse=tf.AUTO_REUSE, auxiliary_name_scope=False) as scope:
    #     with tf.name_scope(scope.original_name_scope) as n:
    #         hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw,
    #                                                                      cell_bw=self.bw,
    #                                                                      inputs=inputs,
    #                                                                      sequence_length=lengths,
    #                                                                      dtype=tf.float32,
    #                                                                      scope=scope)
    #         outputs_concat = tf.concat(hidden_states, axis=2)
    #         return self._reduce_fn(outputs_concat,axis=1)

    with tf.variable_scope("test", auxiliary_name_scope=False) as scope:
      hidden_states, cell_states = tf.nn.dynamic_rnn(cell=self.fw,inputs=inputs,scope=scope,dtype=tf.float32)
      return self._reduce_fn(hidden_states, axis=1)

    # encoder = BiRNN()
    # premise_encoded = encoder(embedded_premise, features["len_prem"])
    # hypothesis_encoded = encoder(embedded_hypothesis, features["len_hyp"], reuse=True)

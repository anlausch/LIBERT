import tensorflow as tf

class DecoderGRUCell(tf.contrib.rnn.GRUCell):

  def __init__(self,
               encoder_state,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(DecoderGRUCell, self).__init__(num_units, activation, reuse, kernel_initializer,
                                         bias_initializer, name) # tanh is default activation
    self._encoder_state = encoder_state
    self.encoder_state_shape = encoder_state[1].shape[0].value

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    self.input_depth = inputs_shape[1].value
    self._gate_kernel = self.add_variable("gates/decoder_kernel",
                                          shape=[self.input_depth +
                                                 self.encoder_state_shape +
                                                 self._num_units, 2*self._num_units],
                                          initializer=self._kernel_initializer)
    self._gate_bias = self.add_variable("gates/decoder_bias",
                                        shape=[2*self._num_units],
                                        initializer=(
                                          self._bias_initializer
                                          if self._bias_initializer is not None
                                          else tf.constant_initializer(1.0, dtype=self.dtype)))
    self._candidate_kernel = self.add_variable("candidate/decoder_kernel",
                                               shape=[self.input_depth +
                                                      self.encoder_state_shape +
                                                      self._num_units, self._num_units],
                                               initializer=self._kernel_initializer)
    self._candidate_bias = self.add_variable("candidate/decoder_bias",
                                             shape=[self._num_units],
                                             initializer=(
                                               self._bias_initializer
                                               if self._bias_initializer is not None
                                               else tf.zeros_initializer(dtype=self.dtype)))
    self.built = True

  def call(self, inputs, state):
    gru_inputs = tf.concat([inputs, state, self._encoder_state], 1)
    gate_inputs = tf.matmul(gru_inputs, self._gate_kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._gate_bias)

    value = tf.sigmoid(gate_inputs)
    r, z = tf.split(value=value, num_or_size_splits=2, axis=1)
    r_state = r * state

    candidate = tf.matmul(tf.concat([inputs, r_state, self._encoder_state],1), self._candidate_kernel)
    candidate = tf.nn.bias_add(candidate, self._candidate_bias)

    h_tilde = self._activation(candidate)
    new_h = (1-z) * state + z * h_tilde
    return new_h, new_h
import config as c
import model.blocks as b
import numpy as np
import os
from util.debug_util import *
from model.cells import DecoderGRUCell
from model import blocks

def _helper(line):
  try:
    parts = line.split(" ")
    emb = parts[len(parts)-300: len(parts)]
    return emb
  except Exception as e:
    return 0 #emb


# glove_numpy_cache="/home/rlitschk/glovenumpycache.npy"
def get_pretrained_embeddings(path): # todo: add vocab size into name
  tgt_path = path + ".npy"
  if not os.path.exists(tgt_path):
    with open(path) as f:
      lines = f.readlines()

    tmp = []
    for line in lines:
      if len(line.split(" ")) == 2:
        continue
      tmp.append(line.replace("\n", "").rstrip())
      if len(tmp) == c.VOCAB_SIZE:
        break
    lines = tmp

    oov_random = np.random.uniform(-0.1, 0.1, size=len(lines[0].split(" "))-1)
    print("pretrained embeddings loaded")
    # pool = multiprocessing.Pool(processes=30)
    result = [oov_random] + list(map(_helper, lines))
    print("embeddings from vocabulary separated")
    result = np.array(result, dtype=np.float32)
    print("embeddings to np.array transformed")
    np.save(tgt_path, result)
    print(result.shape)
    return result
  else:
    return np.load(tgt_path)


def seq_decoder(h_x, vocab_size_tgt, name_scope, word_embeddings_tgt_sentence, sentence_lengths,
                word_indices_target_sentence, hidden_size):
  # gradient clipping probably not needed since we use tanh instead of relu
  # with tf.device('/cpu:0'):y
  with tf.name_scope(name_scope) as ns:
    projection_layer = tf.layers.Dense(vocab_size_tgt, use_bias=False)  # TODO: check on bias
    assert len(h_x.shape) == 2
    decoder_cell = DecoderGRUCell(h_x, hidden_size, name = "rl_gru")
    zeros = tf.get_variable(name_scope + "/initial_state", initializer=tf.zeros_initializer(),
                            shape=[c.BATCH_SIZE, hidden_size])
    helper = tf.contrib.seq2seq.TrainingHelper(word_embeddings_tgt_sentence, sentence_lengths)
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, zeros, output_layer=projection_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=ns)
    logits = outputs.rnn_output

  with tf.variable_scope("masked_crossent"):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=word_indices_target_sentence, logits=logits)
    max_length = tf.reduce_max(sentence_lengths)
    sentence_lengths_transposed = tf.expand_dims(sentence_lengths, axis=1)
    length_range = tf.range(0, max_length, 1)
    range_row = tf.expand_dims(length_range, 0)
    mask = tf.cast(tf.less(range_row, sentence_lengths_transposed), dtype=tf.float32)
    masked_crossent = tf.multiply(crossent, mask)
    loss = tf.div(tf.reduce_sum(masked_crossent), c.BATCH_SIZE, name="crossent_loss")

  return loss


def seq_encoder(inputs, lengths, hidden_size, variable_scope="shared_encoder"):
  """
  Applies and reuses the same encoder model by default. Create private encoder passing value to scope_name parameter
  :param inputs: padded input sequences
  :param lengths: true lengths of padded sequences
  :param variable_scope: variable scope for private encoder
  :param hidden_size: size of hidden states
  :return: sentence representation
  """
  with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
    outputs, output_states = b.biGRU(inputs, dim=hidden_size, seq_len=lengths) # TODO: try out lstm
    # with biGRUs cell_states = hidden_states
    outputs_concat = tf.concat(outputs, axis=2)
    # last_hidden_state = b.last_output(outputs_concat, lengths)
    max_pooled = tf.reduce_max(outputs_concat, axis=1)
    return  max_pooled


def seq_decoder_inference(h_x, vocab_size_tgt, name_scope, embeddings, start_token_id, end_token_id,
                          hidden_size, inference_batch_size = 1, sample = True, max_length = 100):
  with tf.name_scope(name_scope) as ns:
    if sample:
      helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embeddings,
      tf.fill([inference_batch_size], start_token_id), end_token_id)
    else:
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
      tf.fill([inference_batch_size], start_token_id), end_token_id)

    projection_layer = tf.layers.Dense(vocab_size_tgt, use_bias=False)
    assert len(h_x.shape) == 2
    decoder_cell = DecoderGRUCell(h_x, hidden_size, name = "rl_gru")
    zeros = tf.get_variable(name_scope + "/initial_state", initializer=tf.zeros_initializer(),
                            shape=[inference_batch_size, hidden_size])

    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, zeros, output_layer=projection_layer)
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=ns, maximum_iterations = max_length)
    generated = outputs.sample_id
    return generated


def attention_convolutional_encoder(inputs, queries, lengths, mlp_layers=[100, 100], filters={3: 8, 4: 8, 5: 8},
                                    k_max_pools=2, name="att_conv_encoder"):
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    dim_size = tf.reduce_max(lengths)
    inputs = tf.reshape(inputs, [c.BATCH_SIZE, -1, c.EMBEDDING_SIZE])
    lengths = tf.reshape(lengths, [c.BATCH_SIZE])

    all_feats = []
    for fsize in filters:
      slices = blocks.tf_overlap_slice(inputs, 1, np.int64(fsize), dim_size, stride_size=1)
      slices = tf.reshape(slices, [-1, c.BATCH_SIZE, fsize * c.EMBEDDING_SIZE])
      queries_reshaped = tf.tile(tf.expand_dims(tf.reshape(queries, [c.BATCH_SIZE, queries.get_shape()[1]]), 0),
                                 [tf.shape(slices)[0], 1, 1])
      slices_query_concat = tf.transpose(tf.concat((slices, queries_reshaped), axis=2), [1, 0, 2])

      batch = tf.unstack(slices_query_concat, c.BATCH_SIZE)
      ls = tf.unstack(lengths, c.BATCH_SIZE)

      fsize_feats = []
      for ex, l in zip(batch, ls):
        ex_feats = []
        for i in range(filters[fsize]):
          # generates scores for each slice of the input (previously concatenated with query), mlp's are shared/reused (see reuse of dense layers inside of the mlp block)
          ex_slice_outs = blocks.mlp(ex, mlp_layers + [1], name="mlp_size_" + str(fsize) + "_num_" + str(i))

          # zero-out scores originating from padded tokens
          mask_ones = tf.ones(shape=(l - (fsize - 1)), dtype=tf.float32)
          mask_zeros = tf.zeros(shape=(tf.cast(tf.shape(ex_slice_outs)[0], dtype=tf.int64) - (l - (fsize - 1))),
                                dtype=tf.float32)
          mask = tf.concat((mask_ones, mask_zeros), axis=0)
          ex_slice_outs_masked = tf.multiply(ex_slice_outs, tf.reshape(mask, [-1, 1]))

          # max pooling
          vals, _ = tf.nn.top_k(tf.reshape(ex_slice_outs_masked, [-1]), k_max_pools)
          ex_feats.append(vals)
        fsize_feats.append(tf.concat(tuple(ex_feats), axis=0))
      all_feats.append(fsize_feats)

    batch_feats = []
    for i in range(c.BATCH_SIZE):
      batch_feats.append(tf.concat(tuple([x[i] for x in all_feats]), axis=0))

    return tf.stack(batch_feats)
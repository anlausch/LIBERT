"""
Defines model functions to be plugged into tensorflow estimators. When writing your own model
function please make sure you don't create an unnamed tf.metrics.mean metric, since the estimator
creates one internally during evaluation.
"""
import config as c
import model.blocks as b
import pickle
import tensorflow as tf
from model import blocks
from estimator_extension import ExtendedModeKeys
from model.general import seq_encoder
from model.general import attention_convolutional_encoder

def multichannel_model_fn(features, labels, mode, params):
  name_scope = params['name_scope'] if params.get('name_scope', None) else 'treccar_multichannel'
  contrastive = params.get('contrastive', None)
  tf_fixed_qlen = tf.constant(params["fixed_qlen"], dtype=tf.int32)
  fixed_qlen = params["fixed_qlen"]
  tf_fixed_plen = tf.constant(params["fixed_plen"], dtype=tf.int32)
  num_filters = params["num_filters"]
  out_channels = params["out_channels"]
  fixed_plen = params["fixed_plen"]
  k_max = params["k_max_pooling"]
  qlen = tf.cast(features["len_query"], dtype=tf.int32)
  plen = tf.cast(features["len_par_pos"], dtype=tf.int32)
  vocab_id = pickle.load(open("/home/gglavas/data/trec/vocab_id.pickle","rb"))
  vocab_size = tf.constant(len(vocab_id), dtype=tf.int32)

  def encode_query_document(query, qlen, paragraph, plen):
    """
    assumes query matrix to be batch major shaped
    :param query: matrix batch x n_q of vocab id tokens
    :param qlen: true lengths of sequences before padding
    :param paragraph: matrix of batch x n_d of vocab id tokens
    :param plen: true lengths of sequences before padding
    :return: reshaped (padded and/or cut) interaction image
    """
    def pad_or_cut(batched_sequences, length):
      with tf.variable_scope("pad_or_batch"):
        sequence_shape = tf.shape(batched_sequences)
        # n_examples = sequence_shape[0]
        n_tokens = sequence_shape[1]
        dim_mismatch = length - n_tokens
        hack = tf.maximum(0, dim_mismatch) # this is necessary because lambda eval later is not lazy
        expanded_dim_mismatch = tf.expand_dims(tf.stack([0, hack]),0)
        paddings = tf.concat([[[0,0]], expanded_dim_mismatch],0)
        padded_sequence = tf.pad(batched_sequences, paddings, "CONSTANT")
        cut_sequence = batched_sequences[:,:length]
        do_pad = tf.greater(dim_mismatch, 0)
        # do_cut = tf.less(dim_mismatch, 0)
        # do_nothing = tf.equal(dim_mismatch, 0)
        # shaped = tf.case([#(do_nothing, lambda: batched_sequences),
        #                   (do_cut, lambda: cut_sequence),
        #                   (do_pad, lambda: padded_sequence)], default=lambda: batched_sequences)
        shaped = tf.cond(do_pad, true_fn=lambda: padded_sequence, false_fn=lambda: cut_sequence)
        return shaped

    query = pad_or_cut(query, tf_fixed_qlen)
    query = tf.cast(query, tf.int32)
    paragraph = pad_or_cut(paragraph, tf_fixed_plen)
    paragraph = tf.cast(paragraph, tf.int32)

    # TODO: move embedding lookup to input_fn? cf. "Preprocessing on the CPU"
    # TODO: cf. https://www.tensorflow.org/performance/performance_guide
    with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
      tune_embeddings = params["tune_embeddings"] if mode == ExtendedModeKeys.TRAIN else False
      # emb_path = params.get("pretrained_emb", False)
      # if emb_path:
      #   embedding_initializer = get_pretrained_embeddings(emb_path)
      # else:
      #   embedding_initializer = tf.random_uniform(shape=[vocab_size, c.EMBEDDING_SIZE])
      with tf.device("/cpu:0"):
        embeddings = tf.get_variable("en_embeddings",
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     shape=[len(vocab_id)-1, c.EMBEDDING_SIZE],
                                     trainable=tune_embeddings)
        oov_vector = tf.zeros(shape=[1, c.EMBEDDING_SIZE])
        embeddings = tf.concat([oov_vector, embeddings], 0)
        embedded_query = tf.nn.embedding_lookup(embeddings, query)
        embedded_paragraph = tf.nn.embedding_lookup(embeddings, paragraph)

    with tf.variable_scope(name_scope):
      with tf.variable_scope("multichannel_encoding"):
        with tf.variable_scope("image_channel", reuse=tf.AUTO_REUSE):
          # configured_sentence2onehot_fn = partial(b.sentence2onehot, vocab_size=vocab_size, return_dense=True)
          # onehot_query = tf.map_fn(configured_sentence2onehot_fn, (query, qlen), dtype=tf.int32)
          # onehot_doc = tf.map_fn(configured_sentence2onehot_fn, (paragraph, plen), dtype=tf.int32)
          img_rows = []
          return_dense=True
          for i in range(c.BATCH_SIZE):
            onehot_q = tf.cast(b.sentence2onehot((query[i],qlen[i]),
                                                 vocab_size=vocab_size,
                                                 return_dense=return_dense), tf.float32)
            onehot_p = tf.cast(b.sentence2onehot((paragraph[i], plen[i]),
                                                 vocab_size=vocab_size,
                                                 return_dense=return_dense), tf.float32)
            img_rows.append(tf.matmul(onehot_q, onehot_p, a_is_sparse=True, b_is_sparse=True, transpose_b=True))
          onehot_img = tf.stack(img_rows)
          onehot_img = tf.expand_dims(onehot_img, -1)

        with tf.variable_scope("embedding_channel", reuse=tf.AUTO_REUSE):
          embedding_img = tf.matmul(embedded_query, tf.transpose(embedded_paragraph, [0,2,1])) # batch x len_q x len_d
          embedding_img = tf.expand_dims(embedding_img, -1)

        # TODO: go other direction too (p->q)
        with tf.variable_scope("cnn_layer", reuse=tf.AUTO_REUSE):
          multichannel = tf.concat([onehot_img, embedding_img], -1) # batch x n_q x n_d x num_channels
          strides = [1, 1, 1, 1]
          in_channels = 2
          # (kernel name, kernel shape, time steps)
          filter_height = fixed_plen
          unigram_spec = ("unigram_kernel", [1, filter_height, in_channels, out_channels], fixed_qlen)
          bigram_spec = ("bigram_kernel", [2, filter_height, in_channels, out_channels], fixed_qlen-1)
          trigram_spec = ("trigram_kernel", [3, filter_height, in_channels, out_channels], fixed_qlen-2)
          fourgram_spec = ("fougram_kernel", [4, filter_height, in_channels, out_channels], fixed_qlen-3)
          filter_types = [unigram_spec, bigram_spec, trigram_spec, fourgram_spec]

          # can't take k_max when time steps < k
          for filter_type in filter_types:
            assert filter_type[2] >= k_max

          kernel_outputs = []
          for kernel, shape, _ in filter_types:
            for i in range(num_filters):
              ith_kernel = tf.get_variable(name="%s_%s" % (kernel, str(i)),
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           shape=shape)
              # paddings allowes sized filters to have same output size and can be combined, VALID=no padding
              kernel_out = tf.nn.conv2d(multichannel, filter=ith_kernel, padding="VALID", strides=strides)
              kernel_out = tf.squeeze(kernel_out)
              # k_max_pooling
              for channel in range(out_channels):
                kernel_k_max, _ = tf.nn.top_k(kernel_out[:,:,channel], k=k_max, name="%s_max_pool" % str(k_max))
                # TODO: attention layer here combining channels instead of concatenating
                kernel_outputs.append(kernel_k_max)
          cnn_out = tf.concat(kernel_outputs, 1)

    # TODO: stack CNNs + include residual connections + attention
    # add shape information for dense layer to work
    out_shape = num_filters * out_channels * k_max * len(filter_types)
    integrated_qp_representation = tf.reshape(cnn_out, [c.BATCH_SIZE, out_shape])
    return integrated_qp_representation

  if contrastive:
    input_features_pos = encode_query_document(features["query"], qlen, features["par_pos"], plen)
    input_features_neg = encode_query_document(features["query"], qlen, features["par_neg"], plen)
    if contrastive == "raw":
      layers = params["mlp_layers"] + [1]
      pairwise_pos = tf.nn.softmax(b.mlp(input_features_pos, layers, name="output_space"))
      pairwise_neg = tf.nn.softmax(b.mlp(input_features_neg, layers, name="output_space"))
      diff = tf.subtract(pairwise_neg[:,0], pairwise_pos[:,0])
      loss = tf.reduce_mean(diff)
    elif contrastive == "hinge":
      layers = params["mlp_layers"] + [2]
      pairwise_pos = tf.nn.softmax(b.mlp(input_features_pos, layers, name="output_space"))
      pairwise_neg = tf.nn.softmax(b.mlp(input_features_neg, layers, name="output_space"))
      diff = tf.subtract(pairwise_neg[:,0], pairwise_pos[:,0])
      loss = tf.reduce_mean(tf.maximum(tf.constant(1.0, dtype=tf.float32), diff))
    else:
      raise ValueError("Unknown type of contrastive loss")
    labels = tf.ones(c.BATCH_SIZE, dtype=tf.int32)
    predictions = tf.cast(tf.less(diff, 0), dtype=tf.int32, name="predictions")
  else:
    layers = params["mlp_layers"] + [2]
    pointestimate_features = encode_query_document(features["query"], features["len_query"],
                                                   features["paragraph"], features["len_paragraph"])
    logits = b.mlp(pointestimate_features, layers, name="output_space")
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    predictions = tf.argmax(logits)

  with tf.variable_scope("train_accuracy"):
    train_acc = tf.reduce_mean(input_tensor=tf.cast(tf.equal(labels, predictions), dtype=tf.float32),
                               name="value")
    tf.summary.scalar("summary", train_acc)

  if mode == ExtendedModeKeys.EVAL:
    acc = tf.metrics.accuracy(labels=labels, predictions=predictions, name="dev_accuracy")
    eval_metric_ops = {"accuracy": acc}
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

  if mode == ExtendedModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  return loss, {}


def queryconditioned_model_fn(features, labels, mode, params):
  name_scope = params['name_scope'] if params.get('name_scope', None) else 'treccar_model'
  contrastive = params['contrastive'] if params.get('contrastive', None) else False

  # defining the embeddings tensor with pretrained word embeddinfs
  print("Embeddings layer...")
  with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
    prt_embs = params['embeddings']
    embeddings = tf.get_variable("embeddings", initializer=prt_embs, trainable=False)

  # encoding the query with a Bi-LSTM (with max pooling)
  print("Embedding lookup layer...")
  with tf.variable_scope("encoding_layer", reuse=tf.AUTO_REUSE):
    embedded_query = tf.nn.embedding_lookup(embeddings, features["query"], name="embedded_query")

    if contrastive:
      embedded_par_pos = tf.nn.embedding_lookup(embeddings, features["par_pos"], name="embedded_par_pos")
      if mode != ExtendedModeKeys.PREDICT:
        embedded_par_neg = tf.nn.embedding_lookup(embeddings, features["par_neg"], name="embedded_par_neg")
    else:
      embedded_paragraph = tf.nn.embedding_lookup(embeddings, features["paragraph"], name="embedded_paragraph")

  # query-attentive convolution layer (encoding latent features of paragraph(s) with query encoding as input)
  print("Encoding layer...")
  with tf.variable_scope("encoding_layer", reuse=tf.AUTO_REUSE):
    # TODO: change cell definition or model signature
    query_encoded = seq_encoder(embedded_query, features["len_query"], variable_scope="query_encoder_bilstm",
                                cell="LSTM")
    if contrastive:
      par_pos_feats = attention_convolutional_encoder(embedded_par_pos, query_encoded, features["len_par_pos"],
                                                      name="att_conv", mlp_layers=params["mlp_layers"],
                                                      filters=params["filters"], k_max_pools=params["k_max_pools"])
      if mode != ExtendedModeKeys.PREDICT:
        par_neg_feats = attention_convolutional_encoder(embedded_par_neg, query_encoded, features["len_par_neg"],
                                                        name="att_conv", mlp_layers=params["mlp_layers"],
                                                        filters=params["filters"], k_max_pools=params["k_max_pools"])
    else:
      paragraph_feats = attention_convolutional_encoder(embedded_paragraph, query_encoded, features["len_par"],
                                                        name="att_conv", mlp_layers=params["mlp_layers"],
                                                        filters=params["filters"], k_max_pools=params["k_max_pools"])

  # classification layer
  print("Classification layer and losses...")
  with tf.variable_scope("classification_layer_and_loss", reuse=tf.AUTO_REUSE):
    eval_metrics = {}
    if contrastive:
      ### Option 1: prediction is a raw score (mlp gives a scalar for every example), we ask to maximize the difference between pos and neg score
      if params["contrastive_loss"] == "raw":
        par_pos_class = blocks.mlp(par_pos_feats, params["mlp_layers_classifier"] + [1], name="classifier")
        preds = tf.identity(par_pos_class, name="predictions")

        if mode != ExtendedModeKeys.PREDICT:
          par_neg_class = blocks.mlp(par_neg_feats, params["mlp_layers_classifier"] + [1], name="classifier")
          loss = tf.reduce_sum(tf.subtract(par_neg_class, par_pos_class))

          streaming_mean_loss = tf.metrics.mean(loss, name="streaming_loss")
          tf.summary.scalar('stream_loss', streaming_mean_loss[1])

      ### Option 2: 2-way classification for both pos and neg and then hinge loss
      elif params["contrastive_loss"] == "hinge":
        par_pos_class = tf.nn.softmax(
          blocks.mlp(par_pos_feats, params["mlp_layers_classifier"] + [2], name="classifier"))
        preds = tf.identity(par_pos_class[:, 0], name="predictions")

        if mode != ExtendedModeKeys.PREDICT:
          par_neg_class = tf.nn.softmax(
            blocks.mlp(par_neg_feats, params["mlp_layers_classifier"] + [2], name="classifier"))
          diff = tf.subtract(par_pos_class, par_neg_class)[:,
                 0]  # we take only the difference of probabilities for class "1" (index 0)
          loss = tf.reduce_sum(tf.maximum(tf.subtract(tf.constant(1.0, dtype=tf.float32), diff), 0.0))

      else:
        raise ValueError("Unknown type of contrastive loss")
    else:
      par_class = blocks.mlp(paragraph_feats, params["mlp_layers_classifier"] + [2], name="classifier")
      preds = tf.identity(par_class[:, 0], name="predictions")

      if mode != ExtendedModeKeys.PREDICT:
        labs = tf.map_fn(lambda x: tf.cond(tf.equal(x, 1), lambda: tf.constant([1, 0], dtype=tf.float32),
                                           lambda: tf.constant([0, 1], dtype=tf.float32)), labels)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=par_class, labels=labs))

    if mode == ExtendedModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    if mode == ExtendedModeKeys.TRAIN:
      print("Optimizer and minimization...")
      optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics)

    if mode == ExtendedModeKeys.PREDICT:
      print("Predict mode, returning Estimator Spec")
      return tf.estimator.EstimatorSpec(mode, predictions=preds, eval_metric_ops=eval_metrics)


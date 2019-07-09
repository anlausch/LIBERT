import tensorflow as tf
from estimator_extension import ExtendedModeKeys
from model.general import seq_decoder
from model.general import seq_decoder_inference


def gedichte_model_fn(features, labels, mode, params):
  # print("Defining model...")
  with tf.variable_scope("gedichte_lm", reuse=tf.AUTO_REUSE):
    prt_embs = params['embeddings']
    embeddings = tf.get_variable("embeddings", initializer=prt_embs, trainable=False)

    embedded_titles = tf.nn.embedding_lookup(embeddings, features["title"], name="embedded_titles")
    embedded_poems = tf.nn.embedding_lookup(embeddings, features["poem"], name="embedded_poems")

    len_titles = features["len_title"]
    len_poems = features["len_poem"]

    title_encodings = tf.div(tf.reduce_sum(embedded_titles, axis=1), tf.expand_dims(len_titles, -1))

    if mode == ExtendedModeKeys.TRAIN:
      loss = seq_decoder(title_encodings, prt_embs.shape[0], "decoder", embedded_poems, len_poems, features["poem"],
                         params['hidden_size'])
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops={})

    if mode == ExtendedModeKeys.PREDICT:
      generated = seq_decoder_inference(title_encodings, prt_embs.shape[0], "decoder", embeddings,
                                        params["start_token_id"], params["end_token_id"], params['hidden_size'],
                                        sample=True, max_length=params["max_len"])
      tf.estimator.EstimatorSpec(mode, predictions=generated, eval_metric_ops={})
import tensorflow as tf
import config as c
import copy
import model.blocks as b
from util.debug_util import *

from estimator_extension import ExtendedModeKeys
from model.general import seq_encoder
from model.general import seq_decoder
from model.general import get_pretrained_embeddings
from replantio.vocabulary import load_de_vocab
from replantio.vocabulary import load_en_vocab
from replantio.vocabulary import load_fr_vocab



def skipthought_model_fn(features, labels, mode, params):
  en_vocab = load_en_vocab()
  name_scope = params.get('name_scope', 'skipthought_model')
  impose_orthogonality = params.get('impose_orthogonality', False)
  LAMBDA = params.get('lambda', 1.0)
  shared_private = params.get('shared_private', False)

  with tf.name_scope(name_scope):
    with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
      tune_embeddings = params["tune_embeddings"] if mode == ExtendedModeKeys.TRAIN else False
      embeddings = tf.get_variable("en_embeddings",
                                   initializer=tf.random_uniform(
                                     shape=[len(en_vocab), c.EMBEDDING_SIZE], minval=-1, maxval=1),
                                   trainable=tune_embeddings)
      embedded_tgt_sentence = tf.nn.embedding_lookup(embeddings, features["target_sentence"], name="embedd_targ_sent")
      embedded_next_sentence = tf.nn.embedding_lookup(embeddings, features["next_sentence"], name="embedd_next_sent")

    encoded_sentence = seq_encoder(embedded_tgt_sentence,
                                   hidden_size=params['hidden_size'],
                                   lengths=features["length"])

    if shared_private:
      encoded_sentence_public = encoded_sentence
      encoded_sentence_private = seq_encoder(embedded_tgt_sentence,
                                             hidden_size=params['hidden_size'],
                                             lengths=features["length"],
                                             variable_scope="private_encoder_"+name_scope)
      encoded_sentence = tf.concat([encoded_sentence_public, encoded_sentence_private])
      loss = seq_decoder(encoded_sentence, len(en_vocab), name_scope, embedded_next_sentence, None,
                         features["next_sentence"], params['hidden_size'])

      if impose_orthogonality:
        similarity_shared_private = b.tf_cosine(encoded_sentence_public, encoded_sentence_private)
        orthogonality_loss = tf.reduce_mean(similarity_shared_private)
        loss = LAMBDA * loss + (1 - LAMBDA) * orthogonality_loss
    else:
      loss = seq_decoder(encoded_sentence, len(en_vocab), name_scope, embedded_next_sentence, None,
                         features["next_sentence"], params['hidden_size'])

    metrics = {}
    if mode == ExtendedModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == ExtendedModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      # assert len(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, "mean/total:0")) == 0
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    assert mode == ExtendedModeKeys.MULTI_TASK
    return loss, metrics


def nli_model_fn(features, labels, mode, params):
  en_vocab = load_en_vocab()
  name_scope = params['name_scope'] if params.get('name_scope', None) else 'nli_model'
  impose_orthogonality = params['impose_orthogonality'] if params.get('impose_orthogonality', None) else False
  LAMBDA = params['lambda'] if params.get('lambda', None) else 1.0

  with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
    tune_embeddings = params["tune_embeddings"] if mode == ExtendedModeKeys.TRAIN else False
    emb_path = params.get("pretrained_emb", False)
    if emb_path:
      embedding_initializer = get_pretrained_embeddings(emb_path)
    else:
      embedding_initializer = tf.random_uniform(shape=[len(en_vocab), c.EMBEDDING_SIZE], minval=-1, maxval=1)
    embeddings = tf.get_variable("en_embeddings",
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[len(en_vocab), c.EMBEDDING_SIZE],
                                 trainable=tune_embeddings)

  embedded_premise = tf.nn.embedding_lookup(embeddings, features["premise"], name="embedd_premise")
  embedded_hypothesis = tf.nn.embedding_lookup(embeddings, features["hypothesis"], name="embedd_hypothesis")

  premise_encoded = seq_encoder(embedded_premise, hidden_size=params['hidden_size'], lengths=features["len_prem"])
  hypothesis_encoded = seq_encoder(embedded_hypothesis, hidden_size=params['hidden_size'], lengths=features["len_hyp"])

  with tf.variable_scope(name_scope):
    if params.get('shared_private', None):
      premise_encoded_public = premise_encoded
      premise_encoded_private = seq_encoder(embedded_premise,
                                            hidden_size=params['hidden_size'],
                                            lengths=features["len_prem"],
                                            variable_scope="private_encoder_"+name_scope)
      premise_encoded = tf.concat([premise_encoded, premise_encoded_private], axis=1)

      hypothesis_encoded_public = hypothesis_encoded
      hypothesis_encoded_private = seq_encoder(embedded_hypothesis,
                                               hidden_size=params['hidden_size'],
                                               lengths=features["len_hyp"],
                                               variable_scope="private_encoder_"+name_scope)
      hypothesis_encoded = tf.concat([hypothesis_encoded, hypothesis_encoded_private], axis=1)

      if impose_orthogonality:
        premise_similarities = b.tf_cosine(premise_encoded_private, premise_encoded_public)
        hypothesis_similarities = b.tf_cosine(hypothesis_encoded_private, hypothesis_encoded_public)
        similarities = premise_similarities * 0.5 + hypothesis_similarities * 0.5
        orthogonality_loss = tf.reduce_mean(similarities)

      multiplier = 2
    else:
      multiplier = 1

    h = b.hadamard_product(premise_encoded, hypothesis_encoded)

    with tf.name_scope("output_layer"):
      output_layer_kernel = tf.get_variable(name=name_scope+"/output_layer/output_kernel",
                                            initializer=tf.glorot_uniform_initializer(),
                                            shape=[multiplier * 2 * params['hidden_size'] * 4, 3])
      # output_layer_bias = tf.get_variable(name="snli_model/output_layer/output_bias",
      #                                     initializer=tf.glorot_uniform_initializer())
      logits_0 = tf.matmul(h, output_layer_kernel) # TODO: does paper have MLP here?

      # tf.nn.bias_add(logits_0, output_layer_bias)
      # logits = logits + output_layer_bias
      # logits = tf.tanh(logits_0)
      # logits = tf.layers.dense(h, 3, tf.nn.tanh)
      logits = tf.layers.dropout(inputs=logits_0, rate=0.3, training=mode == ExtendedModeKeys.TRAIN)
      predicted_classes = tf.argmax(logits, 1)

    if mode == ExtendedModeKeys.PREDICT:
      predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(crossent, name="crossent_loss")
    if impose_orthogonality:
      loss = LAMBDA * loss + (1 - LAMBDA) * orthogonality_loss
    streaming_accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    # tf.summary.scalar('accuracy', streaming_accuracy[1])

    eval_metrics = {}
    if mode == ExtendedModeKeys.EVAL:
      eval_metrics['accuracy'] = streaming_accuracy
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics)

    if mode == ExtendedModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
      train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    assert mode == ExtendedModeKeys.MULTI_TASK
    return loss, eval_metrics


def nmt_ende_model_fn(features, labels, mode, params):
  updated_params = copy.deepcopy(params)
  updated_params['name_scope'] = "EN-DE_model"
  updated_params['src_lang'] = "en"
  updated_params['tgt_lang'] = "de"
  updated_params['vocab_size_src'] = len(load_en_vocab())
  updated_params['vocab_size_tgt'] = len(load_de_vocab())
  if params.get('pretrained_emb_de', None) and params.get('pretrained_emb_en'):
    updated_params['pretrained_emb_src'] = params['pretrained_emb_en']
    updated_params['pretrained_emb_tgt'] = params['pretrained_emb_de']
  return _wmt_model_fn(features, labels, mode, updated_params)


def nmt_enfr_model_fn(features, labels, mode, params):
  updated_params = copy.deepcopy(params)
  updated_params['name_scope'] = "EN-FR_model"
  updated_params['src_lang'] = "en"
  updated_params['tgt_lang'] = "fr"
  updated_params['vocab_size_src'] = len(load_en_vocab())
  updated_params['vocab_size_tgt'] = len(load_fr_vocab())
  if params.get('pretrained_emb_en') and params.get('pretrained_emb_fr'):
    updated_params['pretrained_emb_src'] = params['pretrained_emb_en']
    updated_params['pretrained_emb_tgt'] = params['pretrained_emb_fr']
  return _wmt_model_fn(features, labels, mode, updated_params)


# inspired by https://github.com/tensorflow/nmt#decoder
def _wmt_model_fn(features, labels, mode, params):
  if mode == ExtendedModeKeys.PREDICT:
    raise NotImplementedError

  if not (params.get('vocab_size_src',None) and params.get('vocab_size_tgt')):
    raise NotImplementedError("This function is not supposed to be called directly")
  # global serializer
  # if not serializer:
  #   serializer = Serializer()

  name_scope = params['name_scope'] if params.get('name_scope',None) else 'nmt_model'
  impose_orthogonality = params['impose_orthogonality'] if params.get('impose_orthogonality', None) else False
  LAMBDA = params['lambda'] if params.get('lambda', None) else 1.0
  shared_private = params.get('shared_private', False)

  # TODO: fix stuff in tensorboard
  with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
    tune_embeddings = params["tune_embeddings"] if mode == ExtendedModeKeys.TRAIN else False
    # embeddings = tf.get_variable("embeddings", initializer=serializer.embeddings, trainable=tune_embeddings)
    emb_path = params.get("pretrained_emb_src", False)
    if emb_path:
      initializer = get_pretrained_embeddings(emb_path)
    else:
      initializer = tf.random_uniform(shape=[params['vocab_size_src'], c.EMBEDDING_SIZE], minval=-1, maxval=1)
    src_embeddings = tf.get_variable(params['src_lang'] + "_embeddings",
                                     initializer=initializer,trainable=tune_embeddings)

    emb_path = params.get("pretrained_emb_tgt", False)
    if emb_path:
      initializer = get_pretrained_embeddings(emb_path)
    else:
      initializer = tf.random_uniform(shape=[params['vocab_size_src'], c.EMBEDDING_SIZE], minval=-1, maxval=1)
    tgt_embeddings = tf.get_variable(params['tgt_lang'] + "_embeddings",
                                     initializer=initializer,
                                     trainable=tune_embeddings)

  with tf.name_scope(name_scope):
    word_indices_source_sentence = features["sentence_1"]
    word_indices_target_sentence = features["sentence_2"]
    len_sent_1 = tf.cast(features["len_sentence_1"],dtype=tf.int32)
    len_sent_2 = tf.cast(features["len_sentence_2"], dtype=tf.int32)
    embedded_src = tf.nn.embedding_lookup(src_embeddings, word_indices_source_sentence, name="embedd_src")
    embedded_tgt = tf.nn.embedding_lookup(tgt_embeddings, word_indices_target_sentence, name="embedd_tgt")

    h_x = seq_encoder(embedded_src, hidden_size=params['hidden_size'], lengths=len_sent_1)

    if shared_private:
      h_x_public = h_x
      h_x_private = seq_encoder(embedded_src, hidden_size=params['hidden_size'],
                                lengths=len_sent_1, variable_scope="private_ecoder_" + name_scope)
      h_x = tf.concat([h_x_private, h_x], axis=1)

      loss = seq_decoder(h_x, params['vocab_size_tgt'], name_scope, embedded_tgt, len_sent_2,
                         word_indices_target_sentence, params['hidden_size'])

      if impose_orthogonality:
        # similarity_shared_private = tf.reduce_sum(tf.multiply(h_x, h_x_private), axis=1)
        similarity_shared_private = b.tf_cosine(h_x_public, h_x_private)
        orthogonality_loss = tf.reduce_mean(similarity_shared_private)
        loss = LAMBDA * loss + (1 - LAMBDA) * orthogonality_loss

      tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
    else:
      loss = seq_decoder(h_x, params['vocab_size_tgt'], name_scope, embedded_tgt, len_sent_2,
                         word_indices_target_sentence, params['hidden_size'])

    metrics = {} # tf.metrics.mean(loss) created internally during estimator.eval, during train loss is reported per batch
    if mode == ExtendedModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == ExtendedModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(params['learning_rate'])
      train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
      # assert len(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES, "mean/total:0")) == 0
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    assert mode == ExtendedModeKeys.MULTI_TASK
    return loss, metrics


def multitask_model_fn(features, labels, mode, params):
  if mode == ExtendedModeKeys.PREDICT:
    raise NotImplementedError

  MODE = ExtendedModeKeys.MULTI_TASK

  nli_features, nli_label = features["nli"]
  nli_loss, nli_eval_metrics = nli_model_fn(nli_features, nli_label, MODE, params)

  nmt_deen_features, nmt_ende_labels = features["nmt_ende"]
  nmt_ende_loss, nmt_deen_eval_metrics = nmt_ende_model_fn(nmt_deen_features, nmt_ende_labels, MODE, params)

  nmt_enfr_features, nmt_enfr_labels = features["nmt_enfr"]
  nmt_enfr_loss, nmt_fren_eval_metrics = nmt_enfr_model_fn(nmt_enfr_features, nmt_enfr_labels, MODE, params)

  with tf.variable_scope("multitask"):
    loss = nli_loss + nmt_ende_loss + nmt_enfr_loss
    highlevel_metrics = {'nli_loss': tf.metrics.mean(nli_loss, name="nli_loss_metric"),
                         'nmt_ende_loss': tf.metrics.mean(nmt_ende_loss, name="nmt_deen_loss_metric"),
                         'nmt_enfr_loss': tf.metrics.mean(nmt_enfr_loss, name="nmt_fren_loss_metric")}

    eval_metric_ops = {**nli_eval_metrics, **nmt_deen_eval_metrics, **highlevel_metrics}
    if mode == ExtendedModeKeys.EVAL:
      return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    assert mode == ExtendedModeKeys.TRAIN
    with tf.variable_scope("gradients"):
      global_step = tf.train.get_global_step()
      optimizer = tf.train.AdamOptimizer(params["learning_rate"])
      multitask_optimizer = tf.contrib.opt.MultitaskOptimizerWrapper(optimizer)
      train_nli = multitask_optimizer.minimize(nli_loss, global_step=global_step, name="nli_optimize")
      train_nmt_deen = multitask_optimizer.minimize(nmt_ende_loss, global_step=global_step, name="nmt_deen_optimize")
      train_nmt_fren = multitask_optimizer.minimize(nmt_enfr_loss, global_step=global_step, name="nmt_fren_optimize")

    with tf.name_scope("sample_task"):
      sampled_task_id = tf.random_uniform(shape=(), minval=1, maxval=4, dtype=tf.int32, name="sampled_task_id")
      # rand_int = tf.Print(rand_int, [rand_int])
      nli_id = tf.constant(1, dtype=tf.int32, name="nli_id")
      nmt_deen_id = tf.constant(2, dtype=tf.int32, name="nmt_deen_id")
      nmt_fren_id = tf.constant(3, dtype=tf.int32, name="nmt_fren_id")
      multitask_train_op = tf.case(pred_fn_pairs={
        tf.equal(sampled_task_id, nli_id): lambda: train_nli,
        tf.equal(sampled_task_id, nmt_deen_id): lambda: train_nmt_deen,
        tf.equal(sampled_task_id, nmt_fren_id): lambda: train_nmt_fren
      }, exclusive=True, name="multitask_train")

    tf.summary.scalar('nli_loss', nli_loss)
    tf.summary.scalar('nmt_ende_loss', nmt_ende_loss)
    tf.summary.scalar('nmt_enfr_loss', nmt_enfr_loss)

  # TODO: improve task sampling visualization + add orthogonality visualization
  # with tf.variable_scope('accuracy'):
  #   tf.summary.scalar('nli_accuracy', )

  # Metrics are reported at the end of training/evaluation, summaries are reported continously
  eval_metric_ops['avg_task_sampled']= tf.metrics.mean(sampled_task_id, name="task_balance_metric")

  tf.summary.scalar('task_sampled', sampled_task_id)
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=multitask_train_op, eval_metric_ops=eval_metric_ops)


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import wn_config as c
import pickle

from pyexpat import features

import modeling
import optimization
import tensorflow as tf
from preprocess_wn import WordNetProcessor
import functools
#from estimator_extension import EvalRoutineCheckpointSaverListener
from conditioned_scope import cond_scope

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file_standard", None,
    "Input TF example files for the standard bert tasks, i.e. masked LM and next sentence prediction (can be a glob or comma separated).")

flags.DEFINE_string(
    "input_file_wn", None,
    "Input TF example files for the wordnet task (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "multitask", False,
    "Run BERT pretraining in multi task learning setting with WN")

flags.DEFINE_string(
    "wn_model_variant", None,
    "Which wordnet model variant to use: BERT, RESCAL")



def model_fn_builder_bert_standard(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == ExtendedModeKeys.TRAIN) or (mode == ExtendedModeKeys.MULTI_TASK_TRAIN)

    with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
      model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

    with tf.name_scope("standard"):
      (masked_lm_loss,
       masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
           bert_config, model.get_sequence_output(), model.get_embedding_table(),
           masked_lm_positions, masked_lm_ids, masked_lm_weights)

      (next_sentence_loss, next_sentence_example_loss,
       next_sentence_log_probs) = get_next_sentence_output(
           bert_config, model.get_pooled_output(), next_sentence_labels)

      total_loss = masked_lm_loss + next_sentence_loss

      tvars = tf.trainable_variables()

      initialized_variable_names = {}
      scaffold_fn = None
      if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:

          def tpu_scaffold():
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

      if mode == ExtendedModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      elif mode == ExtendedModeKeys.EVAL or mode == ExtendedModeKeys.MULTI_TASK_TRAIN or ExtendedModeKeys.MULTI_TASK_EVAL:
        def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                      masked_lm_weights, next_sentence_example_loss,
                      next_sentence_log_probs, next_sentence_labels):
          """Computes the loss and accuracy of the model."""
          masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                           [-1, masked_lm_log_probs.shape[-1]])
          masked_lm_predictions = tf.argmax(
              masked_lm_log_probs, axis=-1, output_type=tf.int32)
          masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
          masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
          masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
          masked_lm_accuracy = tf.metrics.accuracy(
              labels=masked_lm_ids,
              predictions=masked_lm_predictions,
              weights=masked_lm_weights)
          masked_lm_mean_loss = tf.metrics.mean(
              values=masked_lm_example_loss, weights=masked_lm_weights)

          next_sentence_log_probs = tf.reshape(
              next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
          next_sentence_predictions = tf.argmax(
              next_sentence_log_probs, axis=-1, output_type=tf.int32)
          next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
          next_sentence_accuracy = tf.metrics.accuracy(
              labels=next_sentence_labels, predictions=next_sentence_predictions)
          next_sentence_mean_loss = tf.metrics.mean(
              values=next_sentence_example_loss)

          return {
              "masked_lm_accuracy": masked_lm_accuracy,
              "masked_lm_loss": masked_lm_mean_loss,
              "next_sentence_accuracy": next_sentence_accuracy,
              "next_sentence_loss": next_sentence_mean_loss,
          }

        eval_metrics = (metric_fn, [
            masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
            masked_lm_weights, next_sentence_example_loss,
            next_sentence_log_probs, next_sentence_labels
        ])

        if mode == ExtendedModeKeys.EVAL:
          return tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
        elif mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:
          return total_loss, {'masked_lm_example_loss': masked_lm_example_loss,
                              'masked_lm_log_probs': masked_lm_log_probs,
                              'masked_lm_ids': masked_lm_ids,
                              'masked_lm_weights': masked_lm_weights,
                              'next_sentence_example_loss': next_sentence_example_loss,
                              'next_sentence_log_probs': next_sentence_log_probs,
                              'next_sentence_labels': next_sentence_labels} #eval_metrics
        else:
          raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))
      else:
        raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))


  return model_fn


def model_fn_builder_bert_wordnet(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings, multiclass=False, preserve=False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    if not multiclass:
      num_labels = len(WordNetProcessor().get_labels())
    else:
      num_labels = 13


    is_training = (mode == ExtendedModeKeys.TRAIN) or (mode == ExtendedModeKeys.MULTI_TASK_TRAIN)
    #with tf.variable_scope("", reuse=tf.AUTO_REUSE):
    with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
      model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

    with tf.variable_scope("wordnet"):
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("wn_loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            #one_hot_labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
            one_hot_labels = tf.squeeze(tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32))

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            total_loss = tf.reduce_mean(per_example_loss)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    if mode == ExtendedModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == ExtendedModeKeys.EVAL or mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:

      def metric_fn(per_example_loss, label_ids, logits):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(label_ids, predictions)
        loss = tf.metrics.mean(per_example_loss)
        return {
            "wn_accuracy": accuracy,
            "wn_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
      if mode == ExtendedModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      elif mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:
        return total_loss, {'per_example_loss': per_example_loss, 'label_ids': label_ids, 'logits': logits}
      else:
        raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))
    else:
      raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))


  return model_fn


def model_fn_builder_rescal_contrastive(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings, num_relations):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    pos_ent1_ids = features["pos_ent1_ids"]
    pos_ent1_mask = features["pos_ent1_mask"]
    pos_ent2_ids = features["pos_ent2_ids"]
    pos_ent2_mask = features["pos_ent2_mask"]
    pos_rel_id = features["pos_rel_id"]
    neg_ent1_ids = features["neg_ent1_ids"]
    neg_ent1_mask = features["neg_ent1_mask"]
    neg_ent2_ids = features["neg_ent2_ids"]
    neg_ent2_mask = features["neg_ent2_mask"]
    neg_rel_id = features["neg_rel_id"]

    hidden_size = bert_config.hidden_size


    is_training = (mode == ExtendedModeKeys.TRAIN) or (mode == ExtendedModeKeys.MULTI_TASK_TRAIN)

    # Perform embedding lookup on the word ids.
    with tf.variable_scope("shared/bert/embeddings", reuse=tf.AUTO_REUSE):
      (pos_ent1_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=pos_ent1_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)
      (pos_ent2_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=pos_ent2_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)
      (neg_ent1_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=neg_ent1_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)
      (neg_ent2_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=neg_ent2_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)

      # TODO: Which initializer shall I use?
      relation_tensor = tf.get_variable(
        "relation_tensor", [num_relations, hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

      pos_rel_matrices = tf.squeeze(tf.nn.embedding_lookup(relation_tensor, pos_rel_id))
      neg_rel_matrices = tf.squeeze(tf.nn.embedding_lookup(relation_tensor, neg_rel_id))



    with tf.variable_scope("attention"):

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

      attention_vector = tf.get_variable("attention_vector", shape=[hidden_size], dtype=tf.float32)

      # TODO: Take matmul here?
      pos_ent1_attention_scores = tf.tensordot(pos_ent1_embedding_output, attention_vector, axes=[[2], [0]])
      pos_ent1_attention_weights = tf.expand_dims(masked_softmax(pos_ent1_attention_scores, tf.to_float(pos_ent1_mask)), -1)
      pos_ent1s = tf.squeeze(tf.matmul(pos_ent1_embedding_output, pos_ent1_attention_weights, transpose_a=True))

      pos_ent2_attention_scores = tf.tensordot(pos_ent2_embedding_output, attention_vector, axes=[[2], [0]])
      pos_ent2_attention_weights = tf.expand_dims(masked_softmax(pos_ent2_attention_scores, tf.to_float(pos_ent2_mask)), -1)
      pos_ent2s = tf.squeeze(tf.matmul(pos_ent2_embedding_output, pos_ent2_attention_weights, transpose_a=True))

      neg_ent1_attention_scores = tf.tensordot(neg_ent1_embedding_output, attention_vector, axes=[[2], [0]])
      neg_ent1_attention_weights = tf.expand_dims(masked_softmax(neg_ent1_attention_scores, tf.to_float(neg_ent1_mask)), -1)
      neg_ent1s = tf.squeeze(tf.matmul(neg_ent2_embedding_output, neg_ent1_attention_weights, transpose_a=True))

      neg_ent2_attention_scores = tf.tensordot(neg_ent2_embedding_output, attention_vector, axes=[[2], [0]])
      neg_ent2_attention_weights = tf.expand_dims(masked_softmax(neg_ent2_attention_scores, tf.to_float(neg_ent2_mask)), -1)
      neg_ent2s = tf.squeeze(tf.matmul(neg_ent2_embedding_output, neg_ent2_attention_weights, transpose_a=True))

    with tf.variable_scope("scoring_layer"):
      # RESCAL
      # e_1^T * W_R *e_2
      pos_scores = tf.diag_part(tf.matmul(tf.squeeze(tf.matmul(tf.expand_dims(pos_ent1s, -1), pos_rel_matrices, transpose_a=True)), pos_ent2s, transpose_b=True))
      neg_scores = tf.diag_part(
        tf.matmul(tf.squeeze(tf.matmul(tf.expand_dims(neg_ent1s, -1), neg_rel_matrices, transpose_a=True)), neg_ent2s,
                  transpose_b=True))

      # max(0, 1-(pos-neg))
      per_example_loss = tf.math.maximum(tf.zeros_like(pos_scores), tf.math.subtract(tf.ones_like(pos_scores), tf.math.subtract(pos_scores, neg_scores)))
      total_loss = tf.reduce_mean(per_example_loss)


    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    if mode == ExtendedModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == ExtendedModeKeys.EVAL or mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:

      def metric_fn(per_example_loss):
        loss = tf.metrics.mean(per_example_loss)
        return {
            "wn_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss])
      if mode == ExtendedModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      elif mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:
        return total_loss, {'per_example_loss': per_example_loss}
      else:
        raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))
    else:
      raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))


  return model_fn



def model_fn_builder_rescal_lm(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings, num_relations, all_entities_ids, all_entities_masks):
  """Returns `model_fn` closure for TPUEstimator."""

  #TODO: I am stuck here, as the second entity will be
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    # the first entity is given as ids over wp vocab
    ent1_ids = features["ent1_ids"]
    ent1_mask = features["ent1_mask"]

    # the second entity is our true label over the entity vocab (not over the wp vocab)
    label_ent2_ids = features["ent2_ids"]
    label_ent2_mask = features["ent2_mask"]

    # relation id is given over relation vocab
    rel_id = features["rel_id"]

    hidden_size = bert_config.hidden_size

    is_training = (mode == ExtendedModeKeys.TRAIN) or (mode == ExtendedModeKeys.MULTI_TASK_TRAIN)

    # Perform embedding lookup on the word ids.
    with tf.variable_scope("shared/bert/embeddings", reuse=tf.AUTO_REUSE):
      (ent1_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=ent1_ids,
        vocab_size=bert_config.vocab_size,
        embedding_size=hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="word_embeddings",
        use_one_hot_embeddings=use_one_hot_embeddings)

      relation_tensor = tf.get_variable(
        "relation_tensor", [num_relations, hidden_size, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

      rel_matrices = tf.squeeze(tf.nn.embedding_lookup(relation_tensor, rel_id))


    # this is generating a representation of the entity by averaging
    ent1_embedding_output = tf.divide(tf.reduce_sum(tf.multiply(ent1_embedding_output,tf.expand_dims(tf.to_float(ent1_mask), dim=-1)), axis=1),
                                      tf.reshape(tf.reduce_sum(tf.to_float(ent1_mask), axis=1), [-1, 1]))


    # shape is total_num_entities x max_entitites_length
    all_entities_i = tf.constant(all_entities_ids, dtype=tf.int32)
    all_entities_m = tf.constant(all_entities_masks, dtype=tf.int32)

    # look up the current embeddings of the entities and generate
    (all_entities_embeddings, embedding_table) = modeling.embedding_lookup(
      input_ids=all_entities_i,
      vocab_size=bert_config.vocab_size,
      embedding_size=hidden_size,
      initializer_range=bert_config.initializer_range,
      word_embedding_name="word_embeddings",
      use_one_hot_embeddings=use_one_hot_embeddings)

    # shape is total_num_entities x 1
    all_entities_embeddings = tf.divide(tf.reduce_sum(tf.multiply(all_entities_embeddings,tf.expand_dims(tf.to_float(all_entities_m), dim=-1)), axis=1),
                                      tf.reshape(tf.reduce_sum(tf.to_float(all_entities_m), axis=1), [-1, 1]))


    with tf.variable_scope("scoring_layer"):
      # RESCAL
      # e_1^T * W_R *e_2

      # shape is batch_size * total_num_entities
      scores = tf.matmul(tf.squeeze(tf.matmul(tf.expand_dims(ent1_embedding_output, -1), rel_matrices, transpose_a=True)), all_entities_embeddings, transpose_b=True)

      # We apply one more non-linear transformation before the output layer.
      # TODO: this kills the memory
      #scores = tf.layers.dense(
      #   scores,
      #   units=len(all_entities_ids),
      #   activation=modeling.get_activation(bert_config.hidden_act),
      #   kernel_initializer=modeling.create_initializer(
      #     bert_config.initializer_range))
      #scores = modeling.layer_norm(scores)

      # TODO: Do we want a bias?
      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      #output_bias = tf.get_variable(
      #  "output_bias",
      #  shape=[bert_config.vocab_size],
      #  initializer=tf.zeros_initializer())
      #logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
      #logits = tf.nn.bias_add(logits, output_bias)

      # for each example, softmax over the scores from the dense layer
      log_probs = tf.nn.log_softmax(scores, axis=-1)

      #label_ids = tf.reshape(label_ent2_ids, [-1])

      one_hot_labels = tf.one_hot(
        tf.reduce_sum(label_ent2_ids, -1), depth=len(all_entities_ids), dtype=tf.float32)

      # TODO: Check this carefully again
      per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
      loss = tf.reduce_mean(per_example_loss)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)


    if mode == ExtendedModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == ExtendedModeKeys.EVAL or mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:

      def metric_fn(per_example_loss):
        loss = tf.metrics.mean(per_example_loss)
        return {
            "wn_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss])
      if mode == ExtendedModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      elif mode == ExtendedModeKeys.MULTI_TASK_TRAIN or mode == ExtendedModeKeys.MULTI_TASK_EVAL:
        return loss, {'per_example_loss': per_example_loss}
      else:
        raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))
    else:
      raise ValueError("Only TRAIN, EVAL, and MULTITASK modes are supported: %s" % (mode))


  return model_fn



def multitask_model_fn_builder(bert_config, init_checkpoint, learning_rate,
                                 num_train_steps, num_warmup_steps, use_tpu,
                                 use_one_hot_embeddings, wn_model_variant, all_entities_ids, all_entities_masks):

  def model_fn(features, labels, mode, params):
    if mode == ExtendedModeKeys.PREDICT:
      raise NotImplementedError

    if mode == ExtendedModeKeys.TRAIN:
      MODE = ExtendedModeKeys.MULTI_TASK_TRAIN
    elif mode == ExtendedModeKeys.EVAL:
      MODE = ExtendedModeKeys.MULTI_TASK_EVAL


    global_step = tf.train.get_or_create_global_step()
    tf.logging.log_first_n(level=tf.logging.INFO, msg="Global step: %s" % global_step, n=100)

    standard_features = features["standard"]
    wn_features = features["wn"]



    model_fn_standard = model_fn_builder_bert_standard(bert_config, init_checkpoint, learning_rate,
                                   num_train_steps, num_warmup_steps, use_tpu,
                                   use_one_hot_embeddings)
    standard_loss, standard_eval_metrics = model_fn_standard(standard_features, None, MODE, params)

    if wn_model_variant == "BERT" or wn_model_variant == "WN_PAIRS_BINARY":
      model_fn_wn = model_fn_builder_bert_wordnet(bert_config, init_checkpoint, learning_rate,
                                     num_train_steps, num_warmup_steps, use_tpu,
                                     use_one_hot_embeddings)
    elif wn_model_variant == "BERT_MULTICLASS":
      model_fn_wn = model_fn_builder_bert_wordnet(bert_config, init_checkpoint, learning_rate,
                                                  num_train_steps, num_warmup_steps, use_tpu,
                                                  use_one_hot_embeddings,multiclass=True)
    elif wn_model_variant == "RESCAL_LM":
      model_fn_wn = model_fn_builder_rescal_lm(bert_config, init_checkpoint, learning_rate,
                                     num_train_steps, num_warmup_steps, use_tpu,
                                     use_one_hot_embeddings, num_relations=13, all_entities_ids=all_entities_ids, all_entities_masks=all_entities_masks)
    elif wn_model_variant == "RESCAL_CONTRASTIVE":
      model_fn_wn = model_fn_builder_rescal_contrastive(bert_config, init_checkpoint, learning_rate,
                                     num_train_steps, num_warmup_steps, use_tpu,
                                     use_one_hot_embeddings, num_relations=13)
    else:
      raise ValueError("Wordnet model variant can only be BERT or RESCAL")
    wn_loss, wn_eval_metrics = model_fn_wn(wn_features, None, MODE, params)

    ## TODO: I have no idea whether this is correct as we are not using TPU
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    with tf.variable_scope("multitask"):
      loss = standard_loss + wn_loss

      def metric_fn(standard_eval_metrics, wn_eval_metrics):
        # standard task metrics
        standard_masked_lm_log_probs = tf.reshape(standard_eval_metrics['masked_lm_log_probs'],
                                         [-1, standard_eval_metrics['masked_lm_log_probs'].shape[-1]])
        standard_masked_lm_predictions = tf.argmax(
          standard_masked_lm_log_probs, axis=-1, output_type=tf.int32)
        standard_masked_lm_example_loss = tf.reshape(standard_eval_metrics['masked_lm_example_loss'], [-1])
        standard_masked_lm_ids = tf.reshape(standard_eval_metrics['masked_lm_ids'], [-1])
        standard_masked_lm_weights = tf.reshape(standard_eval_metrics['masked_lm_weights'], [-1])
        standard_masked_lm_accuracy = tf.metrics.accuracy(
          labels=standard_masked_lm_ids,
          predictions=standard_masked_lm_predictions,
          weights=standard_masked_lm_weights)
        standard_masked_lm_mean_loss = tf.metrics.mean(
          values=standard_masked_lm_example_loss, weights=standard_masked_lm_weights)

        standard_next_sentence_log_probs = tf.reshape(
          standard_eval_metrics['next_sentence_log_probs'], [-1, standard_eval_metrics['next_sentence_log_probs'].shape[-1]])
        standard_next_sentence_predictions = tf.argmax(
          standard_next_sentence_log_probs, axis=-1, output_type=tf.int32)
        standard_next_sentence_labels = tf.reshape(standard_eval_metrics['next_sentence_labels'], [-1])
        standard_next_sentence_accuracy = tf.metrics.accuracy(
          labels=standard_next_sentence_labels, predictions=standard_next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
          values=standard_eval_metrics['next_sentence_example_loss'])

        # wordnet metrics
        wn_per_example_loss = wn_eval_metrics['per_example_loss']
        if wn_model_variant=="BERT":
          wn_logits = wn_eval_metrics['logits']
          wn_label_ids = wn_eval_metrics['label_ids']
          wn_predictions = tf.argmax(wn_logits, axis=-1, output_type=tf.int32)
          wn_accuracy = tf.metrics.accuracy(wn_label_ids, wn_predictions)
          wn_loss = tf.metrics.mean(wn_per_example_loss)
          return {
            "wn_eval_accuracy": wn_accuracy,
            "wn_eval_loss": wn_loss,
            "standard_masked_lm_accuracy": standard_masked_lm_accuracy,
            "standard_masked_lm_loss": standard_masked_lm_mean_loss,
            "standard_next_sentence_accuracy": standard_next_sentence_accuracy,
            "standard_next_sentence_loss": next_sentence_mean_loss,
          }
        else:
          wn_loss = tf.metrics.mean(wn_per_example_loss)
          return {
            "wn_eval_loss": wn_loss,
            "standard_masked_lm_accuracy": standard_masked_lm_accuracy,
            "standard_masked_lm_loss": standard_masked_lm_mean_loss,
            "standard_next_sentence_accuracy": standard_next_sentence_accuracy,
            "standard_next_sentence_loss": next_sentence_mean_loss,
          }

      if mode == ExtendedModeKeys.EVAL:
        #return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metrics=(metric_fn, [standard_eval_metrics, wn_eval_metrics]),
            scaffold_fn=scaffold_fn)
      assert mode == ExtendedModeKeys.TRAIN

      with tf.name_scope("select_task"):
        upper_bound = tf.constant(2, dtype=tf.int64)
        task_id = tf.mod(global_step, upper_bound)

        # here are the task ids:
        wn_upper_bound = tf.constant(1, dtype=tf.int64, name="wn_upper_bound")




      multitask_train_op = optimization.create_optimizer_multitask(standard_loss=standard_loss, wn_loss=wn_loss, selected_task_id=task_id,
                                                                   wn_upper_bound=wn_upper_bound, init_lr=learning_rate,
                                                                   num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps,
                                                                   use_tpu=use_tpu)

      tf.summary.scalar('standard_loss', standard_loss)
      tf.summary.scalar('wn_loss', wn_loss)

    return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=multitask_train_op,
      eval_metrics=(metric_fn, [standard_eval_metrics, wn_eval_metrics]),
      scaffold_fn=scaffold_fn)

  return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # standard binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder_standard(input_files,
                            max_seq_length,
                            max_predictions_per_seq,
                            is_training,
                            num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.

    d = d.apply(
      tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    d = d.make_one_shot_iterator().get_next()

    features = {
      'input_ids': d['input_ids'],
      'input_mask': d['input_mask'],
      'segment_ids': d['segment_ids'],
      'masked_lm_positions': d['masked_lm_positions'],
      'masked_lm_ids': d['masked_lm_ids'],
      'masked_lm_weights': d['masked_lm_weights'],
      'next_sentence_labels': d['next_sentence_labels']
    }

    return features

  return input_fn


def input_fn_builder_wordnet(input_files,
                            max_seq_length,
                            is_training,
                            num_cpu_threads=4,
                            max_predictions_per_seq=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "label_ids":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    d = d.make_one_shot_iterator().get_next()

    features = {
      'input_ids': d['input_ids'],
      'input_mask': d['input_mask'],
      'segment_ids': d['segment_ids'],
      'label_ids': d['label_ids'],
    }

    return features

  return input_fn



def input_fn_builder_wordnet_identifiable(input_files,
                            max_seq_length,
                            is_training,
                            num_cpu_threads=4,
                            max_predictions_per_seq=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "ent1_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "ent1_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "ent2_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
        "ent2_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
        "rel_id":
            tf.FixedLenFeature([1], tf.int64),
        "label_id":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    d = d.make_one_shot_iterator().get_next()

    features = {
      'ent1_ids': d['ent1_ids'],
      'ent2_ids': d['ent2_ids'],
      'ent1_mask': d['ent1_mask'],
      'ent2_mask': d['ent2_mask'],
      'rel_id': d['rel_id'],
      'label_id': d['label_id'],
    }

    return features

  return input_fn


def input_fn_builder_wordnet_identifiable_paired(input_files,
                            max_seq_length,
                            is_training,
                            num_cpu_threads=4,
                            max_predictions_per_seq=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "pos_ent1_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "pos_ent1_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "pos_ent2_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
        "pos_ent2_mask":
          tf.FixedLenFeature([max_seq_length], tf.int64),
        "pos_rel_id":
            tf.FixedLenFeature([1], tf.int64),
        "neg_ent1_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
        "neg_ent1_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
        "neg_ent2_ids":
        tf.FixedLenFeature([max_seq_length], tf.int64),
        "neg_ent2_mask":
        tf.FixedLenFeature([max_seq_length], tf.int64),
        "neg_rel_id":
        tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))

    d = d.make_one_shot_iterator().get_next()

    features = {
      'pos_ent1_ids': d['pos_ent1_ids'],
      'pos_ent2_ids': d['pos_ent2_ids'],
      'pos_ent1_mask': d['pos_ent1_mask'],
      'pos_ent2_mask': d['pos_ent2_mask'],
      'pos_rel_id': d['pos_rel_id'],
      'neg_ent1_ids': d['neg_ent1_ids'],
      'neg_ent2_ids': d['neg_ent2_ids'],
      'neg_ent1_mask': d['neg_ent1_mask'],
      'neg_ent2_mask': d['neg_ent2_mask'],
      'neg_rel_id': d['neg_rel_id'],
    }

    return features

  return input_fn


def input_fn_builder_multitask(input_files_standard, input_files_wn,
                            max_seq_length,
                            max_predictions_per_seq,
                            is_training,
                            num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  input_function_standard = input_fn_builder_standard(input_files_standard,
                                                  max_seq_length,
                                                  max_predictions_per_seq,
                                                  is_training,
                                                  num_cpu_threads)
  if FLAGS.wn_model_variant == "BERT" or FLAGS.wn_model_variant == "BERT_MULTICLASS" or FLAGS.wn_model_variant == "WN_PAIRS_BINARY":
    input_function_wordnet = input_fn_builder_wordnet(input_files_wn,
                                                    max_seq_length,
                                                    is_training,
                                                    num_cpu_threads)
  elif FLAGS.wn_model_variant == "RESCAL_LM":
    input_function_wordnet = input_fn_builder_wordnet_identifiable(
      input_files=input_files_wn,
      max_seq_length=max_seq_length,
      is_training=is_training, num_cpu_threads=num_cpu_threads)
  elif FLAGS.wn_model_variant == "RESCAL_CONTRASTIVE":
    input_function_wordnet = input_fn_builder_wordnet_identifiable_paired(
      input_files=input_files_wn,
      max_seq_length=max_seq_length,
      is_training=is_training, num_cpu_threads=num_cpu_threads)

  def input_fn(params):
    return {'standard': input_function_standard(params),
            'wn': input_function_wordnet(params)}
  return input_fn




def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main_bert_standard():
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files_standard = []
  for input_pattern in FLAGS.input_file_standard.split(","):
    input_files_standard.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files standard ***")
  for input_file in input_files_standard:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      keep_checkpoint_max=40,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder_bert_standard(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder_standard(
        input_files=input_files_standard,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder_standard(
        input_files=input_files_standard,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


def main_bert_wn(model_variant="BERT", entities_ids=None, entities_masks=None):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files_wn = []
  for input_pattern in FLAGS.input_file_wn.split(","):
    input_files_wn.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files WordNet ***")
  for input_file in input_files_wn:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=40,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  if model_variant == "BERT" or model_variant == "BERT_MULTICLASS":
    model_fn = model_fn_builder_bert_wordnet(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
  elif model_variant == "RESCAL_LM":
    model_fn = model_fn_builder_rescal_lm(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        num_relations=13,
        all_entities_ids=entities_ids,
        all_entities_masks=entities_masks
    )
  elif model_variant == "RESCAL_CONTRASTIVE":
    model_fn = model_fn_builder_rescal_contrastive(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        num_relations=13
    )
  else:
    raise ValueError("Only BERT and RESCAL model variants are supported: %s" % model_variant)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    if model_variant == "BERT" or model_variant == "BERT_MULTICLASS":
      train_input_fn = input_fn_builder_wordnet(
          input_files=input_files_wn,
          max_seq_length=FLAGS.max_seq_length,
          is_training=True)
    elif model_variant == "RESCAL_LM":
      train_input_fn = input_fn_builder_wordnet_identifiable(input_files=input_files_wn,max_seq_length=FLAGS.max_seq_length, is_training=True)
    elif model_variant == "RESCAL_CONTRASTIVE":
      train_input_fn = input_fn_builder_wordnet_identifiable_paired(
      input_files=input_files_wn,
      max_seq_length=FLAGS.max_seq_length,
      is_training=True)
    else:
      raise ValueError("Only BERT and RESCAL model variants are supported: %s" % model_variant)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    if model_variant == "BERT":
      eval_input_fn = input_fn_builder_wordnet(
          input_files=input_files_wn,
          max_seq_length=FLAGS.max_seq_length,
          is_training=False)
    elif model_variant == "RESCAL":
      # eval_input_fn = input_fn_builder_wordnet_identifiable_paired(
      #     input_files=input_files_wn,
      #     max_seq_length=FLAGS.max_seq_length,
      #     is_training=False)
      eval_input_fn = input_fn_builder_wordnet_identifiable(input_files=input_files_wn,
                                                               max_seq_length=FLAGS.max_seq_length, is_training=False)
    else:
      raise ValueError("Only BERT and RESCAL model variants are supported: %s" % model_variant)

    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))


def main_bert_multitask(wn_model_variant="BERT", all_entities_ids=None, all_entities_masks=None):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files_standard = []
  for input_pattern in FLAGS.input_file_standard.split(","):
    input_files_standard.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files standard, i.e. for masked LM and NSP ***")
  for input_file in input_files_standard:
    tf.logging.info("  %s" % input_file)

  input_files_wn = []
  for input_pattern in FLAGS.input_file_wn.split(","):
    input_files_wn.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files WordNet ***")
  for input_file in input_files_wn:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=40,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = multitask_model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      wn_model_variant=wn_model_variant,
      all_entities_ids=all_entities_ids,
      all_entities_masks=all_entities_masks)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

    train_input_fn = input_fn_builder_multitask(
        input_files_standard=input_files_standard,
        input_files_wn=input_files_wn,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)

    # # We removed this, because for the first PoC, we train for a certain number of steps
    # eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=FLAGS.output_dir,
    #                                                         path_eval_script=c.PATH_EVAL_MULTITASK_SCRIPT,
    #                                                         server=c.DEV_SERVER,
    #                                                         gpu_fraction=c.DEV_CUDA_GPU_FRAC,
    #                                                         cuda_visible_devices=c.DEV_CUDA_VISIBLE_DEVICES,
    #                                                         params={})

    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)#,
                    #saving_listeners=[eval_hook_listener], hooks=[eval_hook_listener])

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    eval_input_fn = input_fn_builder_multitask(
        input_files_standard=input_files_standard,
        input_files_wn=input_files_wn,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)


    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))



class ExtendedModeKeys(tf.estimator.ModeKeys):
  MULTI_TASK_TRAIN = "multitask_train"
  MULTI_TASK_EVAL = "multitask_eval"

def main(_):
  if FLAGS.multitask:
    if FLAGS.wn_model_variant == "RESCAL_LM":
      entities_ids = pickle.load(open("./../data/wn_wordpiece_ids.p", "rb"))
      entities_masks = pickle.load(open("./../data/wn_masks.p", "rb"))
      entities_terms = pickle.load(open("./../data/wn_tokens.p", "rb"))
      wn_terms = []
      with open('./../data/wn_terms_vocab.txt') as f:
        for line in f.readlines():
          wn_terms.append(line.strip())
      main_bert_multitask(wn_model_variant=FLAGS.wn_model_variant, all_entities_ids=entities_ids, all_entities_masks=entities_masks)
    elif FLAGS.wn_model_variant == "BERT_MULTICLASS" or FLAGS.wn_model_variant == "BERT" or FLAGS.wn_model_variant == "RESCAL_CONTRASTIVE" or FLAGS.wn_model_variant == "WN_PAIRS_BINARY":
      main_bert_multitask(wn_model_variant=FLAGS.wn_model_variant)
  else:
    main_bert_standard()
    #if FLAGS.wn_model_variant == "RESCAL_LM":
    #  entities_ids = pickle.load(open("./../data/wn_wordpiece_ids.p", "rb"))
    #  entities_masks = pickle.load(open("./../data/wn_masks.p", "rb"))
    #  entities_terms = pickle.load(open("./../data/wn_tokens.p", "rb"))
    #  wn_terms = []
    #  with open('./../data/wn_terms_vocab.txt') as f:
    #    for line in f.readlines():
    #      wn_terms.append(line.strip())
    #  main_bert_wn(model_variant=FLAGS.wn_model_variant, entities_ids=entities_ids, entities_masks=entities_masks)
    #else:
    #  main_bert_wn(model_variant=FLAGS.wn_model_variant)

if __name__ == "__main__":
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

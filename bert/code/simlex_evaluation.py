import codecs
import tokenization
import tensorflow as tf
import modeling
import scipy.stats as stats
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
from tensorflow.contrib.framework.python.framework.checkpoint_utils import *

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
  "init_checkpoint", None,
  "Model checkpoint")

flags.DEFINE_string(
  "bert_config_file", None,
  "The config json file corresponding to the pre-trained BERT model. "
  "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")


def pairwise_eucl_dist(A, B):
  """
  Computes pairwise distances between each elements of A and each elements of B.
  Args:
    A,    [m,d] matrix
    B,    [n,d] matrix
  Returns:
    D,    [m,n] matrix of pairwise distances
  """
  with tf.variable_scope('pairwise_dist'):
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as row and nb as column vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
  return D


def load_simlex(path="/work/anlausch/SimLex-999/SimLex-999.txt"):
  data = []
  with codecs.open(path, "r", "utf8") as f:
    for i, line in enumerate(f.readlines()):
      # omit header
      if i != 0:
        parts = line.split("\t")
        w1 = parts[0]
        w2 = parts[1]
        sim = float(parts[3])
        data.append([w1, w2, sim])
  return data


def tokenize_simlex(data=[], vocab_file="./../data/vocab_extended.txt", max_len=128):
  tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)
  ent1s = []
  ent2s = []
  ys = []
  for d in data:
    ent1s.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d[0])))
    ent2s.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(d[1]))),
    ys.append(d[2])
  lengths_w1 = [len(e) for e in ent1s]
  lengths_w2 = [len(e) for e in ent2s]

  for e1, e2 in zip(ent1s, ent2s):
    while len(e1) < max_len:
      e1.append(0)
    while len(e2) < max_len:
      e2.append(0)
  return ent1s, ent2s, ys, lengths_w1, lengths_w2, max_len


class RescalModel:
  def __init__(self, bilinear_product=True, metric="Euclidean", only_first_token=False):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    with tf.device('/cpu:0'):
      self.ent1 = tf.placeholder(shape=[None, 128], dtype=tf.int32)
      self.ent2 = tf.placeholder(shape=[None, 128], dtype=tf.int32)
      self.ent1_lengths = tf.placeholder(shape=(None,), dtype=tf.int32)
      self.ent2_lengths = tf.placeholder(shape=(None,), dtype=tf.int32)

      ent1_mask = tf.to_float(tf.sequence_mask(self.ent1_lengths, 128))
      ent2_mask = tf.to_float(tf.sequence_mask(self.ent2_lengths, 128))

      with tf.variable_scope("shared/bert/embeddings", reuse=tf.AUTO_REUSE):
        (ent1_embedding_output, embedding_table) = modeling.embedding_lookup(
          input_ids=self.ent1,
          vocab_size=bert_config.vocab_size,
          embedding_size=bert_config.hidden_size,
          initializer_range=bert_config.initializer_range,
          word_embedding_name="word_embeddings",
          use_one_hot_embeddings=False)

        (ent2_embedding_output, embedding_table) = modeling.embedding_lookup(
          input_ids=self.ent2,
          vocab_size=bert_config.vocab_size,
          embedding_size=bert_config.hidden_size,
          initializer_range=bert_config.initializer_range,
          word_embedding_name="word_embeddings",
          use_one_hot_embeddings=False)

        if bilinear_product:
          self.relations = tf.placeholder(shape=(None,), dtype=tf.int32)
          relation_tensor = tf.get_variable(
            "relation_tensor", [13, bert_config.hidden_size, bert_config.hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

          synonymy_matrices = tf.squeeze(tf.nn.embedding_lookup(relation_tensor, self.relations))

      if only_first_token:
        ent1_embedding_output = tf.squeeze(tf.slice(ent1_embedding_output, begin=[0, 0, 0], size=[999, 1, 768]))
        ent2_embedding_output = tf.squeeze(tf.slice(ent2_embedding_output, begin=[0, 0, 0], size=[999, 1, 768]))
        self.normed_e1 = tf.nn.l2_normalize(ent1_embedding_output, axis=1)
        self.normed_e2 = tf.nn.l2_normalize(ent2_embedding_output, axis=1)
        self.scores = tf.diag_part(tf.matmul(self.normed_e1, self.normed_e2, transpose_b=True))

      else:

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

        with tf.variable_scope("attention"):
          attention_vector = tf.get_variable("attention_vector", shape=[bert_config.hidden_size], dtype=tf.float32)
          ent1_attention_scores = tf.tensordot(ent1_embedding_output, attention_vector, axes=[[2], [0]])
          ent1_attention_weights = tf.expand_dims(masked_softmax(ent1_attention_scores, ent1_mask), -1)
          self.ent1_embeddings = tf.squeeze(tf.matmul(ent1_embedding_output, ent1_attention_weights, transpose_a=True))
          ent2_attention_scores = tf.tensordot(ent2_embedding_output, attention_vector, axes=[[2], [0]])
          ent2_attention_weights = tf.expand_dims(masked_softmax(ent2_attention_scores, ent2_mask), -1)
          self.ent2_embeddings = tf.squeeze(tf.matmul(ent2_embedding_output, ent2_attention_weights, transpose_a=True))
        if bilinear_product:
          self.scores = tf.diag_part(
            tf.matmul(tf.squeeze(tf.matmul(tf.expand_dims(self.ent1_embeddings, -1), synonymy_matrices, transpose_a=True)), self.ent2_embeddings,
                      transpose_b=True))
        else:
          self.normed_e1 = tf.nn.l2_normalize(self.ent1_embeddings, axis=1)
          self.normed_e2 = tf.nn.l2_normalize(self.ent2_embeddings, axis=1)
          if metric == "Euclidean":
            self.scores = tf.diag_part(pairwise_eucl_dist(self.ent1_embeddings, self.ent2_embeddings))
          elif metric == "Cosine":
            self.scores = tf.diag_part(tf.matmul(self.normed_e1, self.normed_e2, transpose_b=True))
          else:
            raise ValueError("Only euclidean and cosine supported")


class BertModel:
  '''
  currently only takes the first token representations
  '''
  def __init__(self, average=True):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    with tf.device('/cpu:0'):
      self.ent1 = tf.placeholder(shape=[None, 128], dtype=tf.int32)
      self.ent2 = tf.placeholder(shape=[None, 128], dtype=tf.int32)
      self.ent1_lengths = tf.to_float(tf.placeholder(shape=(None,), dtype=tf.int32))
      self.ent2_lengths = tf.to_float(tf.placeholder(shape=(None,), dtype=tf.int32))

      ent1_mask = tf.to_float(tf.sequence_mask(self.ent1_lengths, 128))
      ent2_mask = tf.to_float(tf.sequence_mask(self.ent2_lengths, 128))

      (ent1_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=self.ent1,
        vocab_size=bert_config.vocab_size,
        embedding_size=bert_config.hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="bert/embeddings/word_embeddings",
        use_one_hot_embeddings=False)

      (ent2_embedding_output, embedding_table) = modeling.embedding_lookup(
        input_ids=self.ent2,
        vocab_size=bert_config.vocab_size,
        embedding_size=bert_config.hidden_size,
        initializer_range=bert_config.initializer_range,
        word_embedding_name="bert/embeddings/word_embeddings",
        use_one_hot_embeddings=False)

      if not average:
        ent1_embedding_output = tf.squeeze(tf.slice(ent1_embedding_output, begin=[0,0,0], size=[999,1,768]))
        ent2_embedding_output = tf.squeeze(tf.slice(ent2_embedding_output, begin=[0,0,0], size=[999,1,768]))
      else:
        ent1_embedding_output = tf.divide(tf.reduce_sum(tf.multiply(ent1_embedding_output, tf.expand_dims(ent1_mask, dim=-1)), axis=1), tf.reshape(self.ent1_lengths, [-1,1]))
        ent2_embedding_output = tf.divide(tf.reduce_sum(tf.multiply(ent2_embedding_output, tf.expand_dims(ent2_mask, dim=-1)), axis=1), tf.reshape(self.ent2_lengths, [-1,1]))
      self.normed_e1 = tf.nn.l2_normalize(ent1_embedding_output, axis=1)
      self.normed_e2 = tf.nn.l2_normalize(ent2_embedding_output, axis=1)
      self.scores = tf.diag_part(tf.matmul(self.normed_e1, self.normed_e2, transpose_b=True))



class ElmoModel:
  def __init__(self):
    with tf.device('/cpu:0'):
      elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

      self.i1 = tf.placeholder(shape=(None,), name="input1", dtype=tf.string)
      self.i2 = tf.placeholder(shape=(None,), name="input2", dtype=tf.string)
      # default: a fixed mean - pooling of all contextualized word representations with shape[batch_size, 1024].
      e1 = elmo(self.i1, signature="default", as_dict=True)["default"]
      e2 = elmo(self.i2, signature="default", as_dict=True)["default"]

      normed_e1 = tf.nn.l2_normalize(e1, axis=1)
      normed_e2 = tf.nn.l2_normalize(e2, axis=1)
      self.scores = tf.diag_part(tf.matmul(normed_e1, normed_e2, transpose_b=True))


def run_rescal(bilinear_product=True, metric="Cosine", only_first_token=False):
  model = RescalModel(bilinear_product=bilinear_product, metric=metric, only_first_token=only_first_token)
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, FLAGS.init_checkpoint)
  tvars = tf.trainable_variables()
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))

  data = load_simlex()
  w1s_tokens, w2s_tokens, ys, lengths_w1, lengths_w2, max_len = tokenize_simlex(data, FLAGS.vocab_file)
  num_examples = len(w1s_tokens)

  if bilinear_product:
    relation_ids = [9 for i in range(num_examples)]

    scores = sess.run(model.scores,{model.ent1: w1s_tokens, model.ent1_lengths: lengths_w1, model.ent2: w2s_tokens,
                                             model.ent2_lengths: lengths_w2, model.relations: relation_ids})
  else:
    scores = sess.run(model.scores,{model.ent1: w1s_tokens, model.ent1_lengths: lengths_w1, model.ent2: w2s_tokens,
                                             model.ent2_lengths: lengths_w2})

  print("Spearman", stats.spearmanr(scores, np.array(ys)).correlation)
  print("Pearson", stats.pearsonr(scores, np.array(ys))[0])


def run_elmo():
  model = ElmoModel()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  data = load_simlex()
  w1s_tokens, w2s_tokens, ys, lengths_w1, lengths_w2, max_len = tokenize_simlex(data, FLAGS.vocab_file)

  w1s = []
  w2s = []
  for d in data:
    w1s.append(d[0])
    w2s.append(d[1])

  scores = sess.run(model.scores, {model.i1: w1s, model.i2: w2s})
  print("Spearman", stats.spearmanr(scores, np.array(ys)).correlation)
  print("Pearson", stats.pearsonr(scores, np.array(ys))[0])


def run_plain_bert(average=True):
  model = BertModel(average=average)
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, FLAGS.init_checkpoint)
  tvars = tf.trainable_variables()
  (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
  tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
  for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
      init_string = ", *INIT_FROM_CKPT*"
    print("name = %s, shape = %s%s" % (var.name, var.shape, init_string))

  data = load_simlex()
  w1s_tokens, w2s_tokens, ys, lengths_w1, lengths_w2, max_len = tokenize_simlex(data, FLAGS.vocab_file)
  scores = sess.run(model.scores, {model.ent1: w1s_tokens, model.ent1_lengths: lengths_w1, model.ent2: w2s_tokens,
                                   model.ent2_lengths: lengths_w2})
  print("Spearman", stats.spearmanr(scores, np.array(ys)).correlation)
  print("Pearson", stats.pearsonr(scores, np.array(ys))[0])

def main():
  run_plain_bert(average=True)
  #run_rescal(bilinear_product=False, metric="Euclidean", only_first_token=True)
  #run_elmo()

if __name__=="__main__":
  main()
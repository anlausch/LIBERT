import config as c
import codecs
import tensorflow as tf
import json
import logging
import numpy as np
import multiprocessing
import os
import re
import pickle
import csv

from collections import namedtuple
from collections import Counter
from functools import partial
from nltk.tokenize import WordPunctTokenizer

import replantio.vocabulary as vocab

store_embeddings = "/home/rlitschk/embeddings/dump.np"
store_vocab_id_dict = "/home/rlitschk/embeddings/vocab_id.pickle"

NMTSpec = namedtuple("Spec", ["src_s1", "src_s2", "tar"])
NLISpec = namedtuple("Spec", ["src_list", "tar"])
TRECCarSpec = namedtuple("Spec", ["rels", "tar"])

nli_train_dev_test = [NLISpec(src_list=[c.snli_train, c.mnli_train], tar=c.nli_tfrecord_train),
                      NLISpec(src_list=[c.snli_dev, c.mnli_dev], tar=c.nli_tfrecord_dev),
                      NLISpec(src_list=[c.snli_test], tar=c.nli_tfrecord_test)]
nmt_enfr_train_dev = [NMTSpec(c.europarl_en, c.europarl_fr, c.nmt_enfr_tfrecord_train),
                      NMTSpec(c.newstest_en, c.newstest_fr, c.nmt_enfr_tfrecord_dev)]

nmt_ende_train_dev = [NMTSpec(c.europarl_en, c.europarl_de, c.nmt_ende_tfrecord_train),
                      NMTSpec(c.newstest_en, c.newstest_de, c.nmt_ende_tfrecord_dev)]

en_vocab = None
fr_vocab = None
de_vocab = None

FORMAT = '%(asctime)-15s %(message)s'
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger()


def _helper(line):
  parts = line.split()
  return ''.join(parts[:len(parts)-300])


def _build_vocab(paths, tokenize_fn, topk=c.VOCAB_SIZE):
  vocab_freq = Counter()
  records = 0
  for path in paths:
    with open(path) as f:
      for i, line in enumerate(f):
        vocab_freq.update(tokenize_fn(line))
      records += i
  if topk:
    vocab_freq = vocab_freq.most_common(topk)
    return [token for token, freq in vocab_freq]
  else:
    return list(vocab_freq.keys())


def build_vocabulary_from_pretrained_emb(path_pretrained_emb, path_vocabulary_store):
  """
  Reads embeddings from @path_pretrained_emb and extracts vocabulary to be stored into @path_vocabulary_store
  :param path_pretrained_emb: points to file containing pretrained embedding
  :param path_vocabulary_store: location where vocabulary file is stored
  :return:
  """
  if os.path.exists(path_vocabulary_store):
    return pickle.load(open(path_vocabulary_store, "rb"))
  if os.path.exists(path_pretrained_emb):
    with open(path_pretrained_emb, "r") as f:
      lines = f.readlines()[:c.VOCAB_SIZE]
    pool = multiprocessing.Pool(processes=20)
    tokens = pool.map(_helper, lines)
    tokens = ["OOV"] + tokens
    with open(path_vocabulary_store, "wb") as f:
      pickle.dump(tokens, f)
    return tokens
  else:
    raise FileNotFoundError("en glove file not found %s" % path_pretrained_emb)


def build_vocabularies():
  logger.info("Start building vocabularies")
  if not os.path.exists(c.PATH_EN_VOCAB):
    global en_vocab
    nli_vocab = SerializerNLI.build_vocab([c.snli_train, c.snli_dev, c.mnli_train, c.mnli_dev])
    nmt_en_vocab = SerializerNMT.build_vocab([c.europarl_en])
    en_vocab = set(nli_vocab + nmt_en_vocab)
    en_vocab = ["OOV", "<s>", "</s>"] + list(en_vocab)
    pickle.dump(en_vocab, open(c.PATH_EN_VOCAB, "wb+"))
    logger.info("english vocabulary built")

  if not os.path.exists(c.PATH_FR_VOCAB):
    global fr_vocab
    fr_vocab = SerializerNMT.build_vocab([c.europarl_fr])
    fr_vocab = ["OOV", "<s>", "</s>"] + fr_vocab
    pickle.dump(fr_vocab, open(c.PATH_FR_VOCAB, "wb+"))
    logger.info("french vocabulary built")

  if not os.path.exists(c.PATH_DE_VOCAB):
    global de_vocab
    de_vocab = SerializerNMT.build_vocab([c.europarl_de])
    de_vocab = ["OOV", "<s>", "</s>"] + de_vocab
    pickle.dump(de_vocab, open(c.PATH_DE_VOCAB, "wb+"))
    logger.info("german vocabulary built")


def extract_word(line):
  # word = line.split()[-300:]
  return line.split()[0]

import multiprocessing
import random
def build_vocabulary_and_records_from_txt(directory, include_filename, tgt_dir, limit):

  def load_emb_vocab():
    with open("/work/gglavas/data/word_embs/yacle/glove/glove.wiki.de.300.vec","r") as f:
      lines = f.readlines()
    pool = multiprocessing.Pool(processes=50)
    return pool.map(extract_word, lines)

  # emb_vocab = load_emb_vocab()
  # emb_vocab = emb_vocab[:limit]

  emb_vocab2id = {}
  vocab2emb = {}
  with open("/work/gglavas/data/word_embs/yacle/glove/glove.wiki.de.300.vec", "r") as f:
    for i, line in enumerate(f):
      term = line.split()[0]
      embedding = line.split()[-300:]
      emb_vocab2id[term]=i
      vocab2emb[term] = embedding
      if i % 1000 == 0:
        print(str(i))

      if i == limit:
        break

  # newline_token = " <NL> "
  files = os.listdir(directory)
  records = [] # TODO: create one training file rightaway
  vocab = Counter()
  for file in files:
    filepath = directory + "/" + file
    if os.path.isfile(filepath):
      file_lines = open(filepath,"r").readlines()
      for line in file_lines:
        # line = line.replace("\t", newline_token)
        line_tokens = wordpunct_tokenize(line)
        vocab.update(line_tokens)
      if include_filename:
        vocab.update(wordpunct_tokenize(file.split(".")[0])) # filename

  START = "<s>"
  END = "</s>"
  NEWLINE = "<NL>"
  SPECIAL_TOKENS = [START, END, NEWLINE]
  vocab = SPECIAL_TOKENS + [token for token, freq in vocab.items() if token in emb_vocab2id]

  emb_matrix = [np.random.uniform(low=-0.01, high=0.01, size=300),
                np.random.uniform(low=-0.01, high=0.01, size=300),
                np.random.uniform(low=-0.01, high=0.01, size=300)] + [vocab2emb[term] for term in vocab[3:]]
  emb_matrix = np.array(emb_matrix)

  vocab2id = {k: v for k, v in zip(vocab, range(len(vocab)))}
  for i, file in enumerate(files):
    filepath = directory + "/" + file
    if os.path.isfile(filepath):
      file_lines = open(filepath, "r").readlines()
      if len(file_lines) > 0:
        file_lines=["\t".join([line.replace("\n","") for line in file_lines])]

      if file_lines[0].split():
        file_lines = file_lines[0].split("\t")

        lines_of_tokens = []
        for line in file_lines:
          token_line = [str(vocab2id[token]) for token in wordpunct_tokenize(line) if token in vocab2id]
          if len(token_line) > 0:
            lines_of_tokens.append(" ".join(token_line))

        text = " 2 ".join(lines_of_tokens) # add newline id
        text = " 0 " + text + " 1 " #add start end ids
        text_tokens = text.split()

        if len(text_tokens) <= 10:
          continue

        title_tokens = [str(vocab2id[token]) for token in wordpunct_tokenize(file.split(".")[0]) if token in vocab2id]
        while len(title_tokens) < 8:
          tmp_id = random.randint(0, len(text_tokens)-1)
          if tmp_id is not 0 and tmp_id is not 1:
            title_tokens.append(str(text_tokens[tmp_id]))

        title = "0 " + " ".join(title_tokens) + " 1"
        records.append(title + "\t" + text + "\t" + str(len(title.split())) + "\t" + str(len(text.split())))

    print(str(i))

  np.save(tgt_dir+"poetry/embedding_matrix.npy", emb_matrix)

  # vocabulary = [token for token, freqv in vocab.most_common(limit)]
  with open(tgt_dir+"poetry/vocab.txt", "w") as f:
    for entry in vocab:
      f.write(entry + "\n")

  with open(tgt_dir+"poetry/train.txt","w") as f:
    for record in records:
      f.write(record + "\n")


# abstract serializer (abstract base class) to serve as an interface for all serializers
# Methods: build_vocab, tokenize, serialize_tfrecord, deserialize_tf_record 
# TODO: to finish
class Serializer:

  def __init__(self, path_emb=c.PATH_GLOVE300):
    np.random.seed(1337)
    # if os.path.exists(store_embeddings) and os.path.exists(store_vocab_id_dict):
    #   self.embeddings = np.load(store_embeddings)
    #   self.vocab_id = pickle.load(open(store_vocab_id_dict,"rb"))
    # else:
    #   oov_emb = np.random.rand(300)
    #   self.vocab_id = {"OOV": 0}
    #   self.embeddings = [oov_emb]
    #   with open(path_emb) as f:
    #     for _id in range(1,200000):
    #       line = f.readline().replace("\n","").split(" ")
    #       self.vocab_id[line[0]] = _id
    #       self.embeddings.append(np.array(line[1:], dtype=np.float32))
    #   self.embeddings = np.array(self.embeddings, dtype=np.float32)
    #   np.save(open(store_embeddings,"wb+"),self.embeddings)
    #   pickle.dump(self.vocab_id,open(store_vocab_id_dict,"wb+"))

  @staticmethod
  def token2id_dict(tokens, vocab_id):
    # return [vocab_id[word] if word in vocab_id else vocab_id["OOV"]
    #         for word in tokens]
    return [vocab_id[word] if word in vocab_id
            else (vocab_id[word.lower()] if word.lower() in vocab_id
                  else vocab_id["OOV"]) for word in tokens]

  @staticmethod
  def token2id_list(tokens, vocab_id):
    # return [vocab_id.index(word) if word in vocab_id else vocab_id.index("OOV")
    #         for word in tokens]
    return [vocab_id.index(word) if word in vocab_id
            else (vocab_id.index(word.lower()) if word.lower() in vocab_id
                  else vocab_id.index("OOV")) for word in tokens]

  @staticmethod
  def tokenize(sentence):
    words = WordPunctTokenizer().tokenize(sentence)
    return ["<s>"] + words + ["</s>"]


class SerializerSkipThought(Serializer):

  def __init__(self, path_vocab, path_emb=c.PATH_GLOVE300):
    super().__init__(path_emb)
    if not os.path.exists(path_vocab):
      raise FileNotFoundError("Call build vocab first and store results in %s" % path_vocab)

    vocab = pickle.load(open(path_vocab, "rb"))
    self.vocab2id = vocab

  @staticmethod
  def build_vocab(paths, topk=c.VOCAB_SIZE):
    return vocab.build_vocab(topk=topk, tokenize_fn=SerializerSkipThought.tokenize, paths=paths)

  def _construct_skipthought_example(self, loaded_example, max_seq_len):
    pass # TODO: implement

  def serialize_tfrecord(self, spec, limit=-1, max_seq_len=None):
    pass # TODO: implement

  @staticmethod
  def deserialize_tfrecord(example_proto):
    pass # TODO: implement


class SerializerNMT(Serializer):

  def __init__(self, path_vocab_src_lan, path_vocab_tgt_lan, path_emb=c.PATH_GLOVE300):
    super().__init__(path_emb)
    if not os.path.exists(path_vocab_src_lan):
      raise FileNotFoundError("Call build vocab first and store results in %s" % path_vocab_src_lan)
    if not os.path.exists(path_vocab_tgt_lan):
      raise FileNotFoundError("Call build vocab first and store results in %s" % path_vocab_tgt_lan)

    src_vocab = pickle.load(open(path_vocab_src_lan,"rb"))
    self.src_vocab2vid = { word: _id for _id, word in enumerate(src_vocab) }

    with open(path_vocab_tgt_lan,"rb") as f:
      tgt_vocab = pickle.load(f)
    self.tgt_vocab2id = { word: _id for _id, word in enumerate(tgt_vocab) }

  @staticmethod
  def build_vocab(paths, topk=c.VOCAB_SIZE):
    return vocab.build_vocab(paths=paths, tokenize_fn=SerializerNMT.tokenize, topk=topk)

  def _construct_nmt_example(self, loaded_example, max_seq_len):
    sentence_1, sentence_2 = loaded_example
    sentence_1 = Serializer.token2id_dict(SerializerNMT.tokenize(sentence_1), self.src_vocab2vid)
    sentence_2 = Serializer.token2id_dict(SerializerNMT.tokenize(sentence_2), self.tgt_vocab2id)

    sentence_1_len = len(sentence_1)
    sentence_2_len = len(sentence_2)
    if max_seq_len is not None:
      sentence_1_len = min(max_seq_len, sentence_1_len)
      sentence_2_len = min(max_seq_len, sentence_2_len)
      sentence_1 = sentence_1[:sentence_1_len]
      sentence_2 = sentence_2[:sentence_2_len]

    ex = tf.train.SequenceExample()
    ex.context.feature["len_sentence_1"].int64_list.value.append(sentence_1_len)
    ex.context.feature["len_sentence_2"].int64_list.value.append(sentence_2_len)

    sentence_1_tokens = ex.feature_lists.feature_list["sentence_1"]
    sentence_2_tokens = ex.feature_lists.feature_list["sentence_2"]

    for token in sentence_1:
      sentence_1_tokens.feature.add().int64_list.value.append(token)

    for token in sentence_2:
      sentence_2_tokens.feature.add().int64_list.value.append(token)

    return ex

  def serialize_tfrecord(self, spec, limit=-1, max_seq_len=None):
    tar = spec.tar
    sentence_language_1 = spec.src_s1
    sentence_language_2 = spec.src_s2

    if os.path.exists(tar):
      logger.info("%s exists already!" % tar)
      return 0

    writer = tf.python_io.TFRecordWriter(tar)
    i=0
    
    with open(sentence_language_1) as f1, open(sentence_language_2) as f2:
      for s1, s2 in zip(f1, f2):
        ex = self._construct_nmt_example((s1,s2), max_seq_len)
        writer.write(ex.SerializeToString())
        if i> 0 and i % 10000 == 0:
          logger.info("%s instances serialized" % str(i))

        if i == limit:
          break
        else:
          i += 1
      writer.close()
    return i

  @staticmethod
  def deserialize_tfrecord(example_proto):
    context_features = {
      "len_sentence_1": tf.FixedLenFeature([], dtype=tf.int64),
      "len_sentence_2": tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
      "sentence_1": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "sentence_2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
      serialized=example_proto,
      context_features=context_features,
      sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features


class SerializerNLI(Serializer):
  LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
  }

  def __init__(self, path_vocab, path_emb=c.PATH_GLOVE300):
    super().__init__(path_emb)
    if not os.path.exists(path_vocab):
      raise FileNotFoundError("Call build vocab first and store results in %s" % path_vocab)

    vocab = pickle.load(open(path_vocab,"rb"))
    self.vocab2id = { word: _id for _id, word in enumerate(vocab) }

  @staticmethod
  def tokenize(sentence):
    string = re.sub(r'\(|\)', '', sentence)
    words = string.split()
    return words

  @staticmethod
  def build_vocab(paths, topk=c.VOCAB_SIZE):
    vocab_freq=Counter()
    records = 0
    for path in paths:
      with open(path,"r") as f:
        for i, line in enumerate(f):
          ex = json.loads(line)
          tokens = SerializerNLI.tokenize(ex.get("sentence1_binary_parse"))
          vocab_freq.update(tokens)
          tokens = SerializerNLI.tokenize(ex.get("sentence2_binary_parse"))
          vocab_freq.update(tokens)
        records += i
    if topk:
      vocab_freq = vocab_freq.most_common(topk)
      return [token for token, freq in vocab_freq]
    else:
      return list(vocab_freq.keys())

  # cf. https://github.com/dennybritz/tf-rnn/blob/master/sequence_example.ipynb
  def _construct_nli_example(self, loaded_example):
    premise = self.token2id_dict(SerializerNLI.tokenize(loaded_example.get("sentence1_binary_parse")), self.vocab2id)
    hypothesis = self.token2id_dict(SerializerNLI.tokenize(loaded_example.get("sentence2_binary_parse")), self.vocab2id)
    l_premise = len(premise)
    l_hypothesis = len(hypothesis)
    label = SerializerNLI.LABEL_MAP[loaded_example.get("gold_label")]

    ex = tf.train.SequenceExample()
    ex.context.feature["len_prem"].int64_list.value.append(l_premise)
    ex.context.feature["len_hyp"].int64_list.value.append(l_hypothesis)
    ex.context.feature["label"].int64_list.value.append(label)

    premise_tokens = ex.feature_lists.feature_list["premise"]
    hypothesis_tokens = ex.feature_lists.feature_list["hypothesis"]

    for token in premise:
      premise_tokens.feature.add().int64_list.value.append(token)

    for token in hypothesis:
      hypothesis_tokens.feature.add().int64_list.value.append(token)

    return ex

  def serialize_tfrecord(self, spec, limit=-1, verbose=True):
    tar = spec.tar
    src_list = spec.src_list

    if os.path.exists(tar):
      print("%s exists already!" % tar)
      return 0

    i=0
    file_handles = []
    writer = tf.python_io.TFRecordWriter(tar)
    for src in src_list:
      file_handles.append(open(src))

    while file_handles:
      idx = np.random.randint(0, len(file_handles))
      random_line = next(file_handles[idx], None)
      if not random_line:
        del file_handles[idx]
        continue

      loaded_example = json.loads(random_line)
      if loaded_example["gold_label"] not in SerializerNLI.LABEL_MAP:
        continue

      example = self._construct_nli_example(loaded_example)
      writer.write(example.SerializeToString())

      i += 1
      if i % 10000 == 0 and verbose:
        logger.info("%s instances serialized" % str(i))

      if i == limit:
        break

    writer.close()
    logger.info("Done! %s instances serialized" % str(i))
    return i

  @staticmethod
  def deserialize_tfrecord(example_proto):
    context_features = {
      "len_prem": tf.FixedLenFeature([], dtype=tf.int64),
      "len_hyp": tf.FixedLenFeature([], dtype=tf.int64),
      "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
      "premise": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "hypothesis": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
      serialized=example_proto,
      context_features=context_features,
      sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features


# Serializer for TRECCar data
class SerializerTRECCAR(Serializer):
  def __init__(self, path_vocab):
    super().__init__()
    if not os.path.exists(path_vocab):
      raise FileNotFoundError("Call build vocab first and store results in %s" % path_vocab)

    self.vocab2id = pickle.load(open(path_vocab, "rb"))

  @staticmethod
  def build_vocab(path="", size=c.EMBEDDING_SIZE, topk=c.VOCAB_SIZE):
    assert path != "" # TODO: add preconfigured/default path here
    return build_vocabulary_from_pretrained_emb(path, emb_size=size, topk=topk)

  @staticmethod
  def token2id_dict(tokens, vocab_id):
    return [
      vocab_id[word] if word in vocab_id else (vocab_id[word.lower()] if word.lower() in vocab_id else vocab_id["OOV"])
      for word in tokens]

  @staticmethod
  def tokenize(sentence):
    words = WordPunctTokenizer().tokenize(sentence)
    return words

  def serialize_tfrecord(self, spec, limitq=-1, max_seq_len_par=None, contrastive=False):
    current_query = ""
    cnt_queries = 0
    query_examples = []

    writer = tf.python_io.TFRecordWriter(spec.tar)

    with codecs.open(spec.rels, encoding='utf8', errors='replace') as f:
      for i, line in enumerate(f):
        example = line.strip().split('\t')
        if len(example) == 4:
          query = example[1].strip().replace("enwiki:", "").replace("%20", " ")
          if query != current_query:
            currq_toks = SerializerTRECCAR.tokenize(current_query)
            records = self._create_examples_for_query_contrastive(currq_toks, query_examples,
                                                                  max_seq_len_par) if contrastive \
              else self._create_examples_for_query_basic(currq_toks, query_examples, max_seq_len_par)
            for r in records:
              writer.write(r.SerializeToString())

            current_query = query
            query_examples = []
            cnt_queries += 1
            if cnt_queries == limitq:
              break
            else:
              query_examples.append((example[2], example[3]))
          else:
            query_examples.append((example[2], example[3]))

      if len(query_examples) > 0 and current_query != "":
        currq_toks = SerializerTRECCAR.tokenize(current_query)
        records = self._create_examples_for_query_contrastive(current_query, query_examples,
                                                              max_seq_len_par) if contrastive else self._create_examples_for_query_basic(
          current_query, query_examples, max_seq_len_par)
        for r in records:
          writer.write(r.SerializeToString())

      writer.close()
    return cnt_queries

  def _create_examples_for_query_basic(self, query_toks, query_examples, max_seq_len_par=None):
    records = []
    for qe in query_examples:
      r = self._create_record_basic(query_toks, qe, max_seq_len_par=max_seq_len_par)
      if r:
        records.append(r)
    return records

  def _create_examples_for_query_contrastive(self, query_toks, query_examples, max_seq_len_par=None):
    records = []
    pos_examples = [qe for qe in query_examples if qe[1] == "1"]
    neg_examples = [qe for qe in query_examples if qe[1] == "0"]
    for pe in pos_examples:
      for ne in neg_examples:
        r = self._create_record_contrastive(query_toks, pe, ne, max_seq_len_par=max_seq_len_par)
        if r:
          records.append(r)
    return records

  def _create_record_basic(self, query_toks, qe, max_seq_len_par=None):
    toks_par = SerializerTRECCAR.tokenize(qe[0])
    if max_seq_len_par is None or len(toks_par) <= max_seq_len_par:
      tokids_par = SerializerTRECCAR.token2id_dict(toks_par, self.vocab2id)
      tokids_query = SerializerTRECCAR.token2id_dict(query_toks, self.vocab2id)

      r = tf.train.SequenceExample()
      r.context.feature["len_query"].int64_list.value.append(len(tokids_query))
      r.context.feature["len_par"].int64_list.value.append(len(tokids_par))
      r.context.feature["label"].int64_list.value.append(int(qe[1]))

      query_seq = r.feature_lists.feature_list["query"]
      par_seq = r.feature_lists.feature_list["paragraph"]

      for tid in tokids_query:
        query_seq.feature.add().int64_list.value.append(tid)

      for tid in tokids_par:
        par_seq.feature.add().int64_list.value.append(tid)

      return r

    else:
      return None

  def _create_record_contrastive(self, query_toks, pe, ne, max_seq_len_par=None):
    toks_par_pos = SerializerTRECCAR.tokenize(pe[0])
    toks_par_neg = SerializerTRECCAR.tokenize(ne[0])

    if max_seq_len_par is None or (len(toks_par_pos) <= max_seq_len_par and len(toks_par_neg) <= max_seq_len_par):
      tokids_par_pos = SerializerTRECCAR.token2id_dict(toks_par_pos, self.vocab2id)
      tokids_par_neg = SerializerTRECCAR.token2id_dict(toks_par_neg, self.vocab2id)
      tokids_query = SerializerTRECCAR.token2id_dict(query_toks, self.vocab2id)

      r = tf.train.SequenceExample()
      r.context.feature["len_query"].int64_list.value.append(len(tokids_query))
      r.context.feature["len_par_pos"].int64_list.value.append(len(tokids_par_pos))
      r.context.feature["len_par_neg"].int64_list.value.append(len(tokids_par_neg))

      query_seq = r.feature_lists.feature_list["query"]
      par_pos_seq = r.feature_lists.feature_list["par_pos"]
      par_neg_seq = r.feature_lists.feature_list["par_neg"]

      for tid in tokids_query:
        query_seq.feature.add().int64_list.value.append(tid)

      for tid in tokids_par_pos:
        par_pos_seq.feature.add().int64_list.value.append(tid)

      for tid in tokids_par_neg:
        par_neg_seq.feature.add().int64_list.value.append(tid)

      return r

    else:
      return None # TODO: exception raise needed here?

  @staticmethod
  def deserialize_tfrecord_basic(example_proto):
    context_features = {
      "len_query": tf.FixedLenFeature([], dtype=tf.int64),
      "len_par": tf.FixedLenFeature([], dtype=tf.int64),
      "label": tf.FixedLenFeature([], dtype=tf.int64),
    }

    sequence_features = {
      "query": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "paragraph": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
      serialized=example_proto,
      context_features=context_features,
      sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features

  @staticmethod
  def deserialize_tfrecord_contrastive(example_proto):
    context_features = {
      "len_query": tf.FixedLenFeature([], dtype=tf.int64),
      "len_par_pos": tf.FixedLenFeature([], dtype=tf.int64),
      "len_par_neg": tf.FixedLenFeature([], dtype=tf.int64),
    }

    sequence_features = {
      "query": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "par_pos": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "par_neg": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
      serialized=example_proto,
      context_features=context_features,
      sequence_features=sequence_features
    )

    features = dict(sequence_parsed, **context_parsed)  # merging the two dicts
    return features

class SerializerArgRec(Serializer):

  # good to know: https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/tf-records.ipynb
  def _construct_argrec_example(self, loaded_example):
    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      value = value if type(value) == list else [value]
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
      """Returns a float_list from a float / double."""
      value = value if type(value) == list else [value]
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      value = value if type(value) == list else [value]
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    feature = {
      'speech_title': _bytes_feature(loaded_example[0].encode('utf-8')),
      'speech_text': _bytes_feature(loaded_example[1].encode('utf-8')),
      'argument_text': _bytes_feature(loaded_example[2].encode('utf-8')),
      'label': _int64_feature(loaded_example[3]),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

  def serialize_tfrecord(self, target="./data/ibm_debater", mode="TRAIN", transcribed_automatically=True, only_argument_title=True, argument_and_title=False):
    # TODO: I have a couple of data specific parameters here that I need to expose
    def load_debater_speeches(basepath=c.PROJECT_HOME + "/data/ibm_debater/speeches/", transcribed_automatically=transcribed_automatically):
      """
      >>> len(load_debater_speeches())
      200
      >>> load_debater_speeches().shape
      (200, 2)
      """
      speeches = []
      if transcribed_automatically:
        path = os.path.join(basepath, "asr.txt")
      else:
        path = os.path.join(basepath, "trs.txt")
      for subdir, dirs, files in os.walk(path):
        for file in files:
          with codecs.open(os.path.join(path, file), 'r', 'utf8') as f:
            speeches.append([file.split(".")[0], f.read()])
      # return structure contains only speechid and text
      return np.array(speeches)

    def load_debater_arguments(path=c.PROJECT_HOME + "/data/ibm_debater/argumentLabels/idebate-points-information.csv"):
      """
      >>> len(load_debater_arguments())
      380
      >>> load_debater_arguments().shape
      (380, 5)
      """
      arguments = []
      with codecs.open(path, 'r', 'utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
          if i != 0:
            arguments.append([row[0], row[2], row[3], row[5], row[
              7]])  # ['motion-id', 'motion-text', 'point-id', 'point-to-motion-polarity', 'point-title-html', 'point-title', 'point-text-html', 'point-text']
      # contains motionid, argid, argtitle, argpolarity, argtext
      return np.array(arguments)

    def is_mentioned(number_of_acceptors, number_of_rejecters):
      if number_of_acceptors > number_of_rejecters:
        return True
      else:
        return False

    def load_debater_arguments_to_speech(
            path=c.PROJECT_HOME + "data/ibm_debater/argumentLabels/points-to-speeches-labeling.csv"):
      """
      >>> len(load_debater_arguments_to_speech())
      756
      """
      mapping = []
      with codecs.open(path, 'r', 'utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
          if i != 0:
            mapping.append([row[0], row[2], row[4], is_mentioned(int(row[5]), int(row[6]))])  # ['motion-id', 'motion-text', 'point-id', 'point-title', 'speech-title', 'number-of-acceptors', 'number-of-rejecters']
      # contains motionid, argumentid, speechtitle, is_mentioned
      return np.array(mapping)

    def load_debater_motions(path=c.PROJECT_HOME + "data/ibm_debater/argumentLabels/motions-information.csv"):
      """
      >>> len(load_debater_motions()[0])
      30
      >>> len(load_debater_motions()[1])
      20
      """
      motions_dev = []
      motions_test = []
      with codecs.open(path, 'r', 'utf8') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
          if i != 0:
            assert row[6] != row[7]
            if row[7] == "TRUE":
              motions_test.append([row[0], row[8], row[9], row[10], row[11]])
            if row[6] == "TRUE":
              motions_dev.append([row[0], row[8], row[9], row[10], row[
                11]])  # ['motion-id', 'motion-text', 'idebate-motion-text', 'category', 'abstract', 'url', 'is-dev', 'is-test', 'speech-title-1', 'rebuttal-speech-title-1', 'speech-title-2', 'rebuttal-speech-title-2']
      # return structures contain #motionid, #speech1title, #speech2title, #speech3title
      return np.array(motions_dev), np.array(motions_test)

    def load_debater_data(speeches_transcribed_automatically=transcribed_automatically):
      """
      >>> dev, test = load_debater_data(); len(dev)
      30
      """
      motions_dev, motions_test = load_debater_motions()
      speeches = load_debater_speeches(transcribed_automatically=speeches_transcribed_automatically)
      arguments = load_debater_arguments()
      arguments_to_speeches = load_debater_arguments_to_speech()
      all_dev = []
      all_test = []
      # starting with the dev data
      for motion in motions_dev:
        motion = motion.tolist()
        motion.append(
          np.array([speech for speech in speeches if speech[0] in [motion[1], motion[2], motion[3], motion[4]]]))
        motion.append(np.array([argument for argument in arguments if argument[0] == motion[0]]))
        motion.append(np.array([ats for ats in arguments_to_speeches if ats[0] == motion[0]]))
        all_dev.append(np.array(motion, dtype=object))

      for motion in motions_test:
        motion = motion.tolist()
        motion.append(
          np.array([speech for speech in speeches if speech[0] in [motion[1], motion[2], motion[3], motion[4]]]))
        motion.append(np.array([argument for argument in arguments if argument[0] == motion[0]]))
        motion.append(np.array([ats for ats in arguments_to_speeches if ats[0] == motion[0]]))
        all_test.append(np.array(motion, dtype=object))
      return np.array(all_dev), np.array(all_test)

    def transform_debater_data_for_input(data, only_argument_title=only_argument_title, argument_and_title=argument_and_title):
      """
      >>> dev, test = load_debater_data(); data = transform_debater_data_for_input(test); len(data[0])
      4
      """
      transformed_data = []
      for motion in data:
        for mapping in motion[7]:
          argument_id = mapping[1]
          speech_title = mapping[2]
          label = [1, 0] if mapping[3] == 'True' else [0, 1]
          for argument in motion[6]:
            if argument_id == argument[1]:
              if only_argument_title:
                argument_text = argument[3]
              elif argument_and_title:
                argument_text = argument[3] + ". " + argument[4]
              else:
                argument_text = argument[4]
              break
          for speech in motion[5]:
            if " - " in speech_title or " - " in speech[0]:
              print("found")

            if speech_title == speech[0]:
              speech_text = speech[1]
              break
          transformed = np.array([speech_title, speech_text, argument_text, label], dtype=object)
          transformed_data.append(transformed)
      return np.array(transformed_data)

    dev, test = load_debater_data(speeches_transcribed_automatically=transcribed_automatically)
    dev = transform_debater_data_for_input(dev, only_argument_title=only_argument_title, argument_and_title=argument_and_title)
    test = transform_debater_data_for_input(test, only_argument_title=only_argument_title, argument_and_title=argument_and_title)
    # here we do another train/dev split for tuning the model
    split_index = -1 * int(0.2 * float(np.shape(dev)[0]))
    train, dev = dev[:split_index], dev[split_index:]

    if mode == "TRAIN":
      data = train
    elif mode == "DEV":
      data = dev
    elif mode == "TEST":
      data = test
    else:
      raise NotImplementedError

    path = target + "_" + mode
    writer = tf.python_io.TFRecordWriter(path)

    if os.path.exists(target):
      print("%s exists already!" % target)
      return 0


    i = 0
    for d in data:
      example = self._construct_argrec_example(d)
      writer.write(example.SerializeToString())

      i += 1
      if i % 10000 == 0:
        logger.info("%s instances serialized" % str(i))

    writer.close()
    logger.info("Done! %s instances serialized" % str(i))
    return i

  @staticmethod
  def deserialize_tfrecord(example_proto):
    # removed again allow_missing=True
    features = {
      'speech_title': tf.FixedLenFeature([], dtype=tf.string),#, allow_missing=True),
      'speech_text': tf.FixedLenFeature([], dtype=tf.string),#, allow_missing=True),
      'argument_text': tf.FixedLenFeature([], dtype=tf.string),#, allow_missing=True),
      'label': tf.FixedLenFeature([2], dtype=tf.int64)
    }

    parsed = tf.parse_single_example(
      serialized=example_proto,
      features=features
    )

    #features['speech_title'] = tf.decode_raw(features['speech_title'], tf.uint8)
    return parsed


def run_nli_serialization(specs, path_emb=None, name="nli"):
  serializer = SerializerNLI(path_emb)
  # TODO: check more closely again if mismatched should be used? (probably doesn't make sense in multitask setup?)
  # multinli test set unavailable, see: https://www.nyu.edu/projects/bowman/multinli/
  num_serialized_examples = 0
  for spec in specs:
    num_serialized_examples += serializer.serialize_tfrecord(spec) # TODO: add max seq length here? needed?
  logger.info("%s examples serialized for %s" % (str(num_serialized_examples), name))
  return num_serialized_examples


def run_argrec_serialization():
  serializer = SerializerArgRec()
  num_serialized_examples = serializer.serialize_tfrecord(mode="TEST", target="./../data/debater_tfrecords", argument_and_title=False, only_argument_title=False, transcribed_automatically=True)
  logger.info("%s examples serialized" % (str(num_serialized_examples)))
  num_serialized_examples = serializer.serialize_tfrecord(mode="TRAIN", target="./../data/debater_tfrecords", argument_and_title=False, only_argument_title=False, transcribed_automatically=True)
  logger.info("%s examples serialized" % (str(num_serialized_examples)))
  num_serialized_examples = serializer.serialize_tfrecord(mode="DEV", target="./../data/debater_tfrecords", argument_and_title=False, only_argument_title=False, transcribed_automatically=True)
  logger.info("%s examples serialized" % (str(num_serialized_examples)))
  return num_serialized_examples


def run_nmt_serialization(specs, path_vocab_src, path_vocab_tgt, name=""):
  serializer = SerializerNMT(path_vocab_src_lan=path_vocab_src, path_vocab_tgt_lan=path_vocab_tgt)
  serialized_examples = 0
  for spec in specs:
    serialized_examples += serializer.serialize_tfrecord(spec, max_seq_len=c.MAX_SEQ_LEN)
  logger.info("%s examples serialized for %s " % (str(serialized_examples), name))
  return serialized_examples



def prepare_all_task_fastText_tfrecords():
  """
  Prepares records for all tasks and builds vocabulary from faiss embeddings
  :return:
  """
  # tgt lang = English
  vocab.build_vocabulary_from_pretrained_emb(path_vocabulary_store=c.PATH_EN_VOCAB,
                                       path_pretrained_emb=c.PATH_FASTTEXT_EN)

  # tgt lang = German
  vocab.build_vocabulary_from_pretrained_emb(path_vocabulary_store=c.PATH_DE_VOCAB,
                                       path_pretrained_emb=c.PATH_FASTTEXT_DE)
  # tgt lang = French
  vocab.build_vocabulary_from_pretrained_emb(path_vocabulary_store=c.PATH_FR_VOCAB,
                                       path_pretrained_emb=c.PATH_FASTTEXT_FR)

  logger.info("faiss vocabularies built")
  run_nli_serialization(nli_train_dev_test, c.PATH_EN_VOCAB, name="nli")
  run_nmt_serialization(nmt_ende_train_dev, c.PATH_EN_VOCAB, c.PATH_DE_VOCAB, name="EN->DE NMT")
  run_nmt_serialization(nmt_enfr_train_dev, c.PATH_EN_VOCAB, c.PATH_FR_VOCAB, name="EN->FR NMT")


def prepare_NLI_glove_tfrecords():
  """
  Prepares records only for NLI and glove
  :return:
  """
  vocab.build_vocabulary_from_pretrained_emb(path_pretrained_emb=c.PATH_GLOVE300,
                                       path_vocabulary_store=c.PATH_EN_VOCAB)
  logger.info("english vocabulary built")
  num_nli_serialized_examples = run_nli_serialization(nli_train_dev_test, c.PATH_EN_VOCAB_GLOVE300)
  logger.info("%s examples serialized for NLI in total" % str(num_nli_serialized_examples))


def prepare_all_task_tfrecords():
  """
  Builds vocabulary from task data
  :return:
  """
  build_vocabularies()
  ende_serialized_examples = run_nmt_serialization(nmt_ende_train_dev,
                                                   path_vocab_src=c.PATH_EN_VOCAB,
                                                   path_vocab_tgt=c.PATH_DE_VOCAB)

  enfr_serialized_examples = run_nmt_serialization(nmt_enfr_train_dev,
                                                   path_vocab_src=c.PATH_EN_VOCAB,
                                                   path_vocab_tgt=c.PATH_FR_VOCAB)

  nli_serialized_examples = run_nli_serialization(nli_train_dev_test)

  logger.info("%s examples serialized for NLI in total" % str(nli_serialized_examples))
  logger.info("%s examples serialized for en-de NMT in total" % str(ende_serialized_examples))
  logger.info("%s examples serialized for en-fr NMT in total" % str(enfr_serialized_examples))
  logger.info("Done - All train/dev/test sets turned into tfrecords!")

# Main serialization function, runs everything -- loads the data, parses the formats and serializes the examples.
# Each program that is training/testing some combinations of datasets should write their own "main" serialization function
def main():
  run_argrec_serialization()
  # prepare_all_task_tfrecords()
  # prepare_NLI_glove_tfrecords()
  # prepare_all_task_fastText_tfrecords()
  # build_vocabulary_and_records_from_txt("/work/rlitschk/data/poetry/all", True, "/home/rlitschk/", 50000)

if __name__ == "__main__":
  main()

from collections import Counter
import os
import pickle
import multiprocessing
import random
import config as c
from nltk.tokenize import wordpunct_tokenize
import numpy as np

def _helper(line):
  parts = line.split()
  return ''.join(parts[:len(parts)-300])


def _extract_word(line):
  # word = line.split()[-300:]
  return line.split()[0]


def build_vocab(paths, tokenize_fn, topk=c.VOCAB_SIZE):
  """
  Builds and returns vocabulary from a list of files where lines are tokenized by the provided
  function tokenize_fn.
  :param paths: list of files
  :param tokenize_fn: tokenizer
  :param topk: limit vocabulary size
  :return: vocabulary
  """
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
    raise FileNotFoundError("File not found %s" % path_pretrained_emb)


# TODO: refactor function or remove it
def build_vocabulary_and_records_from_txt(directory, include_filename, tgt_dir, limit):

  def load_emb_vocab():
    with open("/work/gglavas/data/word_embs/yacle/glove/glove.wiki.de.300.vec","r") as f:
      lines = f.readlines()
    pool = multiprocessing.Pool(processes=50)
    return pool.map(_extract_word, lines)

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


def load_en_vocab():
  global en_vocab
  if not en_vocab:
    en_vocab = pickle.load(open(c.PATH_EN_VOCAB, "rb"))
  return en_vocab


def load_de_vocab():
  global de_vocab
  if not de_vocab:
    de_vocab = pickle.load(open(c.PATH_DE_VOCAB, "rb"))
  return de_vocab


def load_fr_vocab():
  global fr_vocab
  if not fr_vocab:
    fr_vocab = pickle.load(open(c.PATH_FR_VOCAB, "rb"))
  return fr_vocab

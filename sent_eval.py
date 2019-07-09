from util import timer
import config as c
import pprint
import numpy as np
import senteval
import torch
from model.treccar import seq_encoder
from replantio.serializer import load_en_vocab
import tensorflow as tf
import argparse
from util import gputil
import logging

# cf. https://github.com/facebookresearch/SentEval
all_downstream_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                        'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                        'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                        'Length', 'WordContent', 'Depth', 'TopConstituents','BigramShift', 'Tense',
                        'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--cuda_visible_devices", type=str, help="visible GPUs")
parser.add_argument("-f", "--gpu_fraction", type=float, default=1.0)
parser.add_argument("-m", "--model_home", type=str, required=True, help="checkpoint directory of model")
parser.add_argument("-n", "--results_file", type=str, help="file to store results")
parser.add_argument("-l", "--log_level", type=int, default=10, help="log level defaults to Debug")
parser.add_argument("-t", "--tasks", type=str, nargs="+", default=['MR', 'CR','SUBJ','MPQA','TREC', 'MRPC',
                                                                   'SICKEntailment', 'SICKRelatedness'])
args = parser.parse_args()

FORMAT = '%(asctime)-15s %(message)s'
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=args.log_level, format=FORMAT)

logger = logging.getLogger()
if args.results_file:
  model_home = args.model_home if args.model_home[-1] == "/" else args.model_home + "/"
  results_file = model_home + args.results_file
  fileHandler = logging.FileHandler(filename=results_file)
  fileHandler.setFormatter(formatter)
  logger.addHandler(fileHandler)

assert len(args.tasks) > 0  # at least one task need to be specified
assert all(map(lambda t: t in all_downstream_tasks, args.tasks))  # all tasks must be available
arguments = vars(args)
logger.info("Evaluating SentEval with %s" % vars(args))

# gputil.disable_all_gpus()
gpu_options = gputil.set_gpus_from_args(args)

# http://forums.fast.ai/t/how-to-check-your-pytorch-keras-is-using-the-gpu/7232
# TODO: mode encoder to params?
# TODO: include einstein eval in another file to compare if we improve in one of the two?
def prepare(params, samples):
  pass

pinputs = tf.placeholder(shape=[None,None], dtype=tf.int32)
sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32)
en_vocab = load_en_vocab()
word2id = {vocab: i for i, vocab in enumerate(en_vocab)}

with tf.device('/cpu:0'):
  embedding_matrix = tf.get_variable("embedding_layer/en_embeddings",
                                     initializer=tf.random_uniform(
                                       shape=[len(en_vocab), c.EMBEDDING_SIZE], minval=-1, maxval=1),
                                     trainable=False)
  embedded_sentence = tf.nn.embedding_lookup(embedding_matrix, pinputs, name="embedding_layer/embedd_premise")
  encoded_sentences = seq_encoder(embedded_sentence, sentence_lengths)

ckpt = tf.train.latest_checkpoint(checkpoint_dir=args.model_home)
saver = tf.train.Saver()

config = tf.ConfigProto(gpu_options=gpu_options) if gpu_options else None
sess = tf.Session(config=config)
saver.restore(sess, ckpt)

# gputil.set_gpus_from_args(args)

def batcher(params, batch):
  maxlen = max(map(len, batch))
  lengths = []
  sentences = []

  for sent in batch:
    word_ids = [word2id.get(word, word2id.get("OOV")) for word in sent]
    lengths.append(len(sent))

    while len(word_ids) < maxlen:
      word_ids.append(0)

    sentences.append(word_ids)

  sentences = np.array(sentences, dtype=np.int32)
  lengths = np.array(lengths, dtype=np.int32)
  embeddings = sess.run(encoded_sentences, {pinputs: sentences, sentence_lengths: lengths})
  return embeddings


def prototyping_config():
  params = {'task_path': c.PATH_SENTEVAL_DATA, 'usepytorch': True, 'kfold': 5,
            'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                           'tenacity': 3, 'epoch_size': 2}}
  return params


def default_config():
  params = {'task_path': c.PATH_SENTEVAL_DATA, 'usepytorch': True, 'kfold': 10,
            'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                           'tenacity': 5, 'epoch_size': 4}}
  return params


if __name__ == "__main__":
  timer = timer.Timer()
  # prepare = lambda params, samples: None
  se = senteval.engine.SE(default_config(), batcher, prepare)
  results = se.eval(args.tasks)
  logger.info(pprint.pprint(results))
  duration = timer.pprint_stop()
  logger.info("total duration: %s" % duration)

# TODO: logging util, then tain log, eval log, senteval log
# TODO: continue https://github.com/facebookresearch/SentEval/blob/master/examples/bow.py
# TODO: look at https://einstein.ai/research/the-natural-language-decathlon
# https://www.thegeekstuff.com/linux-101-hacks-ebook/

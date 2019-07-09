import tensorflow as tf
import numpy as np
import json
from replantio.serializer import SerializerNLI
from replantio.serializer import SerializerNMT
from replantio.serializer import SerializerTRECCAR
from replantio.serializer import SerializerArgRec
import config as c
import os
import codecs
import csv


def _get_dataset(paths, is_train, epochs, deserialize_fn):
  dataset = tf.data.TFRecordDataset(paths)
  if is_train:
    dataset = dataset.shuffle(2048)
    dataset = dataset.repeat(epochs)
    dataset = dataset.map(deserialize_fn, num_parallel_calls=32)
  else:
    dataset = dataset.repeat(1)
    dataset = dataset.map(deserialize_fn)
  return dataset


def get_poetry_data(path, epochs=-1, is_train=False, batch_size=c.BATCH_SIZE, cut_final_batch=False):

  def poem_generator(p):
    # max seq. len still missing
    with open(p, "r") as file:
      for line in file:
        title, text, len_title, len_text = line.replace("\n","").split("\t")
        yield len_title, len_text, title.split(), text.split()
  configured_fn = partial(poem_generator, path=path)
  dataset = tf.data.Dataset.from_generator(configured_fn, output_types=(tf.int32,tf.int32, tf.int32, tf.int32))
                                           # output_shapes=(tf.TensorShape([]),
                                           #                tf.TensorShape([]),
                                           #                tf.TensorShape([None]),
                                           #                tf.TensorShape([None])))
  dataset = dataset.repeat(100)
  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(tf.TensorShape([]),
                                                               tf.TensorShape([]),
                                                               tf.TensorShape([None]),
                                                               tf.TensorShape([None])))
  dataset = dataset.make_one_shot_iterator().get_next()
  return dataset


def get_argrec_data_tfrecords(path, epochs=-1, is_train=True, batch_size=c.BATCH_SIZE, cut_final_batch=False):
  dataset = _get_dataset(paths=[path], epochs=epochs, is_train=is_train,
                         deserialize_fn=SerializerArgRec.deserialize_tfrecord)
  #dataset = dataset.repeat(epochs)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.make_one_shot_iterator().get_next()
  features = {
    'speech_title': dataset["speech_title"],
    'speech': dataset["speech_text"],
    'argument': dataset["argument_text"],
    'label': dataset["label"]
  }
  return features, dataset["label"]


def get_argrec_data(path, epochs=-1, is_train=False, batch_size=c.BATCH_SIZE, cut_final_batch=False):
  """
  Contains a couple of helpers + the generator
  """
  # TODO: I have a couple of data specific parameters here that I need to expose
  def load_debater_speeches(basepath=c.PROJECT_HOME + "/data/ibm_debater/speeches/", transcribed_automatically=True):
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

  def is_mentioned(number_of_acceptors, number_of_rejecters, high_confidence=False):
    if high_confidence:
      if number_of_acceptors >= 4:
        return True
      else:
        return False
    else:
      if number_of_acceptors > number_of_rejecters:
        return True
      else:
        return False

  def load_debater_arguments_to_speech(path=c.PROJECT_HOME + "/data/ibm_debater/argumentLabels/points-to-speeches-labeling.csv",
                                       high_confidence=False):
    """
    >>> len(load_debater_arguments_to_speech())
    756
    """
    mapping = []
    with codecs.open(path, 'r', 'utf8') as f:
      reader = csv.reader(f, delimiter=',')
      for i, row in enumerate(reader):
        if i != 0:
          mapping.append([row[0], row[2], row[4], is_mentioned(int(row[5]), int(row[6]),
                                                               high_confidence)])  # ['motion-id', 'motion-text', 'point-id', 'point-title', 'speech-title', 'number-of-acceptors', 'number-of-rejecters']
    # contains motionid, argumentid, speechtitle, is_mentioned
    return np.array(mapping)

  def load_debater_motions(path=c.PROJECT_HOME + "/data/ibm_debater/argumentLabels/motions-information.csv"):
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

  def load_debater_data(speeches_transcribed_automatically=True, high_confidence=False):
    """
    >>> dev, test = load_debater_data(); len(dev)
    30
    """
    motions_dev, motions_test = load_debater_motions()
    speeches = load_debater_speeches(transcribed_automatically=speeches_transcribed_automatically)
    arguments = load_debater_arguments()
    arguments_to_speeches = load_debater_arguments_to_speech(high_confidence=high_confidence)
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

  def transform_debater_data_for_input(data, only_argument_title=True, argument_and_title=False):
    """
    >>> dev, test = load_debater_data(); data = transform_debater_data_for_input(test); len(data[0])
    4
    """
    transformed_data = []
    for motion in data:
      for mapping in motion[7]:
        argument_id = mapping[1]
        speech_title = mapping[2]
        label = [1,0] if mapping[3] == 'True' else [0,1]
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

  def ibm_data_generator(path):
    # TODO: parameterize all this
    dev, test = load_debater_data(speeches_transcribed_automatically=True,
                                         high_confidence=False)
    dev = transform_debater_data_for_input(dev, only_argument_title=False, argument_and_title=False)
    test = transform_debater_data_for_input(test, only_argument_title=False, argument_and_title=False)
    # here we do another train/dev split for tuning the model
    split_index = -1 * int(0.2 * float(np.shape(dev)[0]))
    train, dev = dev[:split_index], dev[split_index:]

    if path == "TRAIN":
      for d in train:
        # speech_title, speech_text, argument_text, label
        yield d[0], d[1], d[2], d[3]
    elif path == "DEV":
      for d in dev:
        # speech_title, speech_text, argument_text, label
        yield d[0], d[1], d[2], d[3]
    elif path == "TEST":
      for d in test:
        # speech_title, speech_text, argument_text, label
        yield d[0], d[1], d[2], d[3]
    else:
      raise NotImplementedError

  from functools import partial
  configured_fn = partial(ibm_data_generator, path=path)
  dataset = tf.data.Dataset.from_generator(configured_fn, output_types=(tf.string, tf.string, tf.string, tf.int32))

  dataset = dataset.repeat(epochs)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.make_one_shot_iterator().get_next()
  speech_title, speech_text, argument_text, label = dataset
  features = {
    'speech_title': speech_title,
    'speech': speech_text,
    'argument': argument_text,
    'label': label
  }
  return features, label



def get_random_snli_data():

  def random_generator():
    file_path = "/work/rlitschk/data/snli_1.0/snli_1.0_train.jsonl"
    # file_path = "/home/rlitschk/data/snli_1.0/snli_1.0_train.jsonl"
    with open(file_path) as f:
      next(f) # skip header
      for line in f:
        # _length = np.random.randint(low=3,high=30)
        length = len(json.loads(line)["sentence1_binary_parse"].split(" "))
        premise = length, np.random.randint(low=-10,high=10,size=(length,30))
        # length2 = np.random.randint(low=3, high=30)
        length2 = len(json.loads(line)["sentence2_binary_parse"].split(" "))
        hypothesis = length2, np.random.randint(low=-10,high=10,size=(length2,30))
        yield np.random.randint(low=0,high=2), premise, hypothesis

  g = random_generator()
  next(g)

  dataset = tf.data.Dataset.from_generator(random_generator, (tf.int32,
                                                              (tf.int32, tf.float32),
                                                              (tf.int32, tf.float32)))
  dataset = dataset.repeat(100)
  dataset = dataset.padded_batch(batch_size=20, padded_shapes=(tf.TensorShape([]),
                                                               (tf.TensorShape([]),[None,30]),
                                                               (tf.TensorShape([]),[None,30])))
  dataset = dataset.make_one_shot_iterator().get_next()
  label, premises, hypotheses = dataset
  len_prem, embedded_premise = premises
  len_hyp, embedded_hypothesis = hypotheses

  return (len_prem, embedded_premise, len_hyp, embedded_hypothesis), label


def get_nmt_data(path, epochs=-1, is_train=False, batch_size=c.BATCH_SIZE, cut_final_batch=False):
  dataset = _get_dataset(paths=[path], epochs=epochs, is_train=is_train,
                         deserialize_fn=SerializerNMT.deserialize_tfrecord)
  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={'len_sentence_1': tf.TensorShape([]),
                                                                       'len_sentence_2': tf.TensorShape([]),
                                                                       'sentence_1': tf.TensorShape([None]),
                                                                       'sentence_2': tf.TensorShape([None])})
  if cut_final_batch: # assumes batch-major shape
    dataset = dataset.filter(lambda x: tf.equal(tf.shape(x['sentence_1'])[0], batch_size))

  dataset = dataset.prefetch(2)
  dataset = dataset.make_one_shot_iterator().get_next()
  features = {
    'len_sentence_1': dataset['len_sentence_1'],
    'len_sentence_2': dataset['len_sentence_2'],
    'sentence_1': dataset['sentence_1'],
    'sentence_2': dataset['sentence_2']
  }
  return features, None


def get_nli_data(path, epochs=-1, is_train=False, batch_size=c.BATCH_SIZE, cut_final_batch=False):
  dataset = _get_dataset(paths=[path], epochs=epochs, is_train=is_train,
                         deserialize_fn=SerializerNLI.deserialize_tfrecord)
  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes={'label' : tf.TensorShape([]),
                                                                       'hypothesis': tf.TensorShape([None]),
                                                                       'premise': tf.TensorShape([None]),
                                                                       'len_hyp': tf.TensorShape([]),
                                                                       'len_prem': tf.TensorShape([])})
  if cut_final_batch:  # assumes batch-major shape
    dataset = dataset.filter(lambda x: tf.equal(tf.shape(x['hypothesis'])[0], batch_size))

  dataset = dataset.prefetch(2)
  dataset = dataset.make_one_shot_iterator().get_next()
  label = dataset['label']
  features = {
    'premise': dataset['premise'],
    'hypothesis': dataset['hypothesis'],
    'len_prem': dataset['len_prem'],
    'len_hyp': dataset['len_hyp'],
    'label': dataset['label']
  }
  return features, label


def get_treccar_data(path, contrastive=False, epochs=-1, is_train=False, batch_size=c.BATCH_SIZE,
                     cut_final_batch=False):
  dataset = _get_dataset(paths=[path], epochs=epochs, is_train=is_train,
                         deserialize_fn=SerializerTRECCAR.deserialize_tfrecord_contrastive
                         if contrastive else SerializerTRECCAR.deserialize_tfrecord_basic)

  if contrastive:
    padded_shapes = {'len_query': tf.TensorShape([]), 'len_par_pos': tf.TensorShape([]),
                     'len_par_neg': tf.TensorShape([]), 'query': tf.TensorShape([None]),
                     'par_pos': tf.TensorShape([None]), 'par_neg': tf.TensorShape([None])}

  else:
    padded_shapes = {'len_query': tf.TensorShape([]), 'len_par': tf.TensorShape([]),
                     'label': tf.TensorShape([]), 'query': tf.TensorShape([None]),
                     'paragraph': tf.TensorShape([None])}

  dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)

  if cut_final_batch:  # assumes batch-major shape
    dataset = dataset.filter(lambda x: tf.equal(tf.shape(x['query'])[0], batch_size))

  dataset = dataset.prefetch(5)
  dataset = dataset.make_one_shot_iterator().get_next()
  if contrastive:
    features = {
      'len_query': dataset['len_query'],
      'len_par_pos': dataset['len_par_pos'],
      'len_par_neg': dataset['len_par_neg'],
      'query': dataset['query'],
      'par_pos': dataset['par_pos'],
      'par_neg': dataset['par_neg']
    }
  else:
    features = {
      'len_query': dataset['len_query'],
      'len_par': dataset['len_par'],
      'label': dataset['label'],
      'query': dataset['query'],
      'paragraph': dataset['paragraph']
    }
    label = dataset['label']

  return features, None if contrastive else label


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = ""

  path_train_wmt_records  = c.PATH_TF_RECORDS + "nmt_data.tfrecord"
  path_train_nli_records = c.PATH_TF_RECORDS + "snli_train.tfrecord"
  e=c.EPOCHS
  it = True

  res = {"nmt": get_nmt_data(path_train_wmt_records, e, it, cut_final_batch=False),
         "nli": get_nli_data(path_train_nli_records, e, it, cut_final_batch=False)}

  path_train_records = c.PATH_TF_RECORDS + "nmt_data.tfrecord"
  path_test_records = c.PATH_TF_RECORDS + "nmt_test_data.tfrecord"
  rand_int = tf.random_uniform(shape=(), minval=1, maxval=3, dtype=tf.int32)
  from functools import partial
  get_train_data = partial(get_nmt_data,
                           path=path_train_records,
                           is_train=True,
                           epochs=c.EPOCHS,
                           cut_final_batch=True)

  from util.debug_util import f
  f(res["nli"])
  f(res["nmt"][0])
  pass

import codecs
import os
import tokenization
import tensorflow as tf
import random
from shutil import copyfile
import collections
import pickle
import util.utils as utils
import util.batch_helper as batch_helper
import numpy as np
import itertools
import math

random.seed = 1024

class InputExample(object):

  def __init__(self, guid, ent_a=None, ent_b=None, rel=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.ent_a = ent_a
    self.ent_b = ent_b
    self.rel = rel
    self.label = label

class InputPair(object):

  def __init__(self, guid, ent_a=None, ent_b=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.ent_a = ent_a
    self.ent_b = ent_b
    self.label = label

class InputExamplePaired(object):

  def __init__(self, guid, triple_pos=None, triple_neg=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.triple_pos = triple_pos
    self.triple_neg = triple_neg


  def to_string(self):
    return str(self.guid) + ", " + str(self.triple_pos) + ", " + str(self.triple_neg)


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id


class InputFeaturesIdentifiable(object):
  """A single set of features of data."""

  def __init__(self, ent1_ids, ent1_mask, ent2_ids, ent2_mask, rel_id, label_id):
    self.ent1_ids = ent1_ids
    self.ent1_mask = ent1_mask
    self.ent2_ids = ent2_ids
    self.ent2_mask = ent2_mask
    self.rel_id = rel_id
    self.label_id = label_id


class InputFeaturesIdentifiablePaired(object):
  """A single set of features of data."""

  def __init__(self, pos_ent1_ids, pos_ent1_mask, pos_ent2_ids, pos_ent2_mask, pos_rel_id, neg_ent1_ids, neg_ent1_mask, neg_ent2_ids, neg_ent2_mask, neg_rel_id):
    self.pos_ent1_ids = pos_ent1_ids
    self.pos_ent1_mask = pos_ent1_mask
    self.pos_ent2_ids = pos_ent2_ids
    self.pos_ent2_mask = pos_ent2_mask
    self.pos_rel_id = pos_rel_id
    self.neg_ent1_ids = pos_ent1_ids
    self.neg_ent1_mask = pos_ent1_mask
    self.neg_ent2_ids = pos_ent2_ids
    self.neg_ent2_mask = pos_ent2_mask
    self.neg_rel_id = pos_rel_id


class WordNetProcessor:
    # def get_train_examples(self, data_dir):
    #   """See base class."""
    #   return self._create_examples(
    #       self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    #
    # def get_dev_examples(self, data_dir):
    #   """See base class."""
    #   return self._create_examples(
    #       self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    #
    # def get_test_examples(self, data_dir):
    #   """See base class."""
    #   return self._create_examples(
    #       self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _load_wn(self, read_path='./../data/WN', filter_vocab=True, vocab=None):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(filter_vocab=True, vocab=tokenization.load_vocab("./../data/vocab_word_level_extended.txt") ); len(wn_triples)
        718015
        """
        wn_terms = {}
        with codecs.open(os.path.join(read_path, 'wn_terms.dict'), 'r') as f:
            for line in f.readlines():
                term, id = line.strip().split('\t')
                if filter_vocab:
                    oov = False
                    tokens = term.split('_')
                    for token in tokens:
                        if token not in vocab:
                            oov = True
                    if not oov:
                        wn_terms[id] = term
                else:
                    wn_terms[id] = term
                f.close()
        wn_rels = {}
        with codecs.open(os.path.join(read_path, 'wn_rels.dict'), 'r') as f:
            for line in f.readlines():
                rel, id = line.strip().split('\t')
                wn_rels[id] = rel
                f.close()
        wn_triples = []
        with codecs.open(os.path.join(read_path, 'wn_triples.txt'), 'r') as f:
            for line in f.readlines():
                e1, e2, rel = line.strip().split('\t')
                if filter_vocab:
                    if e1 in wn_terms and e2 in wn_terms:
                        wn_triples.append([e1, e2, rel])
                else:
                    wn_triples.append([e1, e2, rel])
                f.close()

        print("Size of terms dictionary: %d" % len(wn_terms))
        print("Number of triples: %d" % len(wn_triples))
        return wn_terms, wn_rels, wn_triples


    def _triple_exists(self, ent1, ent2, rel, wn_triples):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        WordNetProcessor()._triple_exists(wn_triples[100][0], wn_triples[100][1], wn_triples[100][2], wn_triples)
        True
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        WordNetProcessor()._triple_exists(wn_triples[100][0], wn_triples[101][1], wn_triples[100][2], wn_triples)
        False
        """
        for triple in wn_triples:
            e1 = triple[0]
            e2 = triple[1]
            r = triple[2]
            if ent1 == e1 and ent2 == e2 and rel == r:
                return True
        return False


    def _create_negative_examples(self, wn_terms, wn_rels, wn_triples, count):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        true_triples, false_triples = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, wn_triples, 100); \
        len(false_triples)
        100
        """
        # first mark all the positive examples as positive
        for triple in wn_triples:
            triple.append(True)

        # now let's create some bullshit
        false_triples = []
        for i, triple in enumerate(wn_triples):
            if i == count:
                break
            ent1 = triple[0]
            ent2 = triple[1]
            rel = triple[2]
            # as long as we have not found a second entity which does not stand in rel with ent1 do
            while self._triple_exists(ent1, ent2, rel, wn_triples) or self._triple_exists(ent1, ent2, rel, false_triples) or ent1 == ent2:
                # choose random entity
                ent2, term = random.choice(list(wn_terms.items()))
            false_triples.append((ent1, ent2, rel, False))
        return wn_triples, false_triples


    def _create_negative_examples_paired(self, wn_terms, wn_rels, wn_triples):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        all_triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples[:100]); \
        len(all_triples)
        100
        """
        # first mark all the positive examples as positive
        for triple in wn_triples:
            triple.append(True)

        # now let's create some bullshit
        all_triples_paired = []
        false_triples = []
        for i, triple in enumerate(wn_triples):
            ent1 = triple[0]
            ent2 = triple[1]
            rel = triple[2]
            # as long as we have not found a second entity which does not stand in rel with ent1 do
            # TODO: Think of the transitive closure problem
            while self._triple_exists(ent1, ent2, rel, wn_triples) or self._triple_exists(ent1, ent2, rel, false_triples) or ent1 == ent2:
                # choose random entity
                ent2, term = random.choice(list(wn_terms.items()))
            false_triples.append((ent1, ent2, rel, False))
            all_triples_paired.append([triple, [ent1, ent2, rel, False]])
        return all_triples_paired


    def extend_vocab_with_rels(self, wn_rels, old_vocab_path='./../data/BERT_base_new/vocab.txt', new_vocab_path='./../data/BERT_base_new/vocab_extended.txt', add_special_tokens=False):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(filter_vocab=False); WordNetProcessor().extend_vocab_with_rels(wn_rels, add_special_tokens=False)
        """
        copyfile(old_vocab_path, new_vocab_path)
        vocab = tokenization.load_vocab(new_vocab_path)
        with codecs.open(new_vocab_path, 'a', 'utf8') as f:
            for key, value in wn_rels.items():
                if value not in vocab:
                    f.write(value)
                    f.write('\n')
                else:
                    print("rel already in")
            if add_special_tokens:
                for value in ["[SEP]", "[CLS]", "[UNK]", "[PAD]"]:
                    if value not in vocab:
                        f.write(value)
                        f.write('\n')
                    else:
                        print("special token already in")
            f.close()


    def _join_wn_information(self, wn_terms, wn_rels, triples):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, wn_triples); \
        len(triples_explicit) == len(wn_triples)
        True
        """
        triples_explicit = []
        try:
            for triple in triples:
                triple_explicit = [wn_terms[triple[0]], wn_terms[triple[1]], wn_rels[triple[2]]]
                if len(triple) == 4:
                    triple_explicit.append(triple[3])
                triples_explicit.append(triple_explicit)
        except Exception as e:
            print(e)
        return triples_explicit


    def _join_wn_information_paired(self, wn_terms, wn_rels, triples):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        all_triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples[:100]); \
        triples_explicit = WordNetProcessor()._join_wn_information_paired(wn_terms, wn_rels, all_triples); \
        len(triples_explicit) == len(all_triples)
        True
        """
        triples_explicit = []
        try:
            for triple_pos, triple_neg in triples:
                triple_explicit = [[wn_terms[triple_pos[0]], wn_terms[triple_pos[1]], wn_rels[triple_pos[2]]], [wn_terms[triple_neg[0]], wn_terms[triple_neg[1]], wn_rels[triple_neg[2]]]]
                if len(triple_pos) == 4:
                    triple_explicit[0].append(triple_pos[3])
                    triple_explicit[1].append(triple_neg[3])
                triples_explicit.append(triple_explicit)
        except Exception as e:
            print(e)
        return triples_explicit


    def _create_examples(self, lines, set_type):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        triples_pos, triples_neg = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, wn_triples, 100); \
        triples = triples_pos + triples_neg; \
        triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, triples); \
        examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN"); \
        len(examples) == len(triples_explicit)
        True
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            ent_a = tokenization.convert_to_unicode(line[0])
            ent_b = tokenization.convert_to_unicode(line[1])
            rel = tokenization.convert_to_unicode(line[2])
            if len(line) > 3:
                label = line[3]
            else:
                label = 0
            examples.append(InputExample(guid=guid, ent_a=ent_a, ent_b=ent_b, rel=rel, label=label))
        return examples


    def _create_examples_paired(self, lines, set_type):
        """
        >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
        triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples[:100]); \
        triples_explicit = WordNetProcessor()._join_wn_information_paired(wn_terms, wn_rels, triples); \
        examples = WordNetProcessor()._create_examples_paired(triples_explicit, "TRAIN"); \
        len(examples) == len(triples_explicit)
        True
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            triple_pos = line[0]
            triple_neg = line[1]
            triple_pos = '; '.join(triple_pos[:3])
            triple_neg = '; '.join(triple_neg[:3])
            triple_pos = tokenization.convert_to_unicode(triple_pos)
            triple_neg = tokenization.convert_to_unicode(triple_neg)
            examples.append(InputExamplePaired(guid=guid, triple_pos=triple_pos, triple_neg=triple_neg))
        return examples

def create_input_pair(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        ent_a = tokenization.convert_to_unicode(line[0])
        ent_b = tokenization.convert_to_unicode(line[1])
        label = line[2]
        examples.append(InputPair(guid=guid, ent_a=ent_a, ent_b=ent_b, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`.
    >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
    triples_pos, triples_neg = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, wn_triples, 100); \
    triples = triples_pos + triples_neg; \
    triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, triples); \
    examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN"); \
    features = [convert_single_example(i, example, WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)) for i, example in enumerate(examples)]; \
    feature = convert_single_example(1, examples[0], WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)); \
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.ent_a)
    tokens_b = tokenizer.tokenize(example.ent_b)
    rel = example.rel

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], ent1, [SEP], ent2, [SEP], rel, [SEP] with "- 5"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 5)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # this is our first entity
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    # this is the second entity
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    # TODO: Eventually we want to add '2' as segment id; also we might want to remove the underscores in the entity names?
    # this is the relation
    tokens.append(rel)
    segment_ids.append(2)
    tokens.append("[SEP]")
    segment_ids.append(2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_single_pair(ex_index, example, label_list, max_seq_length, tokenizer):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.ent_a)
    tokens_b = tokenizer.tokenize(example.ent_b)

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # [CLS] ent1, [SEP] ent2 [SEP]
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # this is our first entity
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    # this is the second entity
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_single_example_relation_classification(ex_index, example, label_list, max_seq_length,
                           bert_tokenizer, rel_tokenizer, extended_segments):
    """Converts a single `InputExample` into a single `InputFeatures`.
    """
    # TODO: Working on this now for prio 1
    # WORKS on original vocab + relation vocab as labels
    tokens_a = bert_tokenizer.tokenize(example.ent_a)
    tokens_b = bert_tokenizer.tokenize(example.ent_b)
    label_rel = [example.rel]

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], ent1, [SEP], ent2, [SEP], with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

    # this is our first entity
    tokens = []
    segment_ids = []
    if extended_segments:
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(2)
        tokens.append("[SEP]")
        segment_ids.append(2)

        # this is the second entity
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(3)
        tokens.append("[SEP]")
        segment_ids.append(3)
    else:
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        # this is the second entity
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    # TODO: we might want to remove the underscores in the entity names?

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = rel_tokenizer.convert_tokens_to_ids(label_rel)

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %s)" % (example.label, str(label_id)))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


def convert_single_example_identifiable(ex_index, example, label_list, max_seq_length,
                           bert_tokenizer, wn_tokenizer, label_entities_tokenizer):
    # TODO: Adapt test
    """Converts a single `InputExample` into a single `InputFeatures`.
    >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
    triples_pos, triples_neg = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, wn_triples, 100); \
    triples = triples_pos + triples_neg; \
    triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, triples); \
    examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN"); \
    features = [convert_single_example(i, example, WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)) for i, example in enumerate(examples)]; \
    feature = convert_single_example(1, examples[0], WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)); \
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = bert_tokenizer.tokenize(example.ent_a)
    if label_entities_tokenizer:
        tokens_b = [example.ent_b]
    else:
        tokens_b = bert_tokenizer.tokenize(example.ent_b)
    rel = [example.rel]

    entity1_ids = bert_tokenizer.convert_tokens_to_ids(tokens_a)
    if label_entities_tokenizer:
        entity2_ids = label_entities_tokenizer.convert_tokens_to_ids(tokens_b)
    else:
        entity2_ids = bert_tokenizer.convert_tokens_to_ids(tokens_b)
    if wn_tokenizer:
        rel_id = wn_tokenizer.convert_tokens_to_ids(rel)
    else:
        bert_tokenizer.convert_tokens_to_ids(rel)

    entity1_mask = [1] * len(entity1_ids)
    entity2_mask = [1] * len(entity2_ids)

    # Zero-pad up to the sequence length.
    while len(entity1_ids) < max_seq_length:
        entity1_ids.append(0)
        entity1_mask.append(0)
    while len(entity2_ids) < max_seq_length:
        entity2_ids.append(0)
        entity2_mask.append(0)

    assert len(entity1_ids) == max_seq_length
    assert len(entity2_ids) == max_seq_length
    assert len(entity1_mask) == max_seq_length
    assert len(entity2_mask) == max_seq_length
    assert len(rel_id) == 1

    label_id = label_map[example.label]

    feature = InputFeaturesIdentifiable(
        ent1_ids=entity1_ids,
        ent1_mask=entity1_mask,
        ent2_ids=entity2_ids,
        ent2_mask=entity2_mask,
        rel_id=rel_id,
        label_id=label_id)
    return feature


def convert_single_example_identifiable_paired(ex_index, example, label_list, max_seq_length,
                                        bert_tokenizer, wn_tokenizer):
    # TODO: ADAPT TEST
    """Converts a single `InputExample` into a single `InputFeatures`.
    >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
    triples_pos, triples_neg = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, wn_triples, 100); \
    triples = triples_pos + triples_neg; \
    triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, triples); \
    examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN"); \
    features = [convert_single_example(i, example, WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)) for i, example in enumerate(examples)]; \
    feature = convert_single_example(1, examples[0], WordNetProcessor().get_labels(), 50, tokenizer = tokenization.FullTokenizer( \
    vocab_file='./../vocab_extended.txt', do_lower_case=False)); \
    """

    triple_pos = example.triple_pos.split("; ")
    triple_neg = example.triple_neg.split("; ")
    assert len(triple_pos) == 3
    assert len(triple_neg) == 3
    pos_ent_a, pos_ent_b, pos_rel = triple_pos
    neg_ent_a, neg_ent_b, neg_rel = triple_neg
    pos_rel = [pos_rel]
    neg_rel = [neg_rel]

    tokens_pos_ent_a = bert_tokenizer.tokenize(pos_ent_a)
    tokens_pos_ent_b = bert_tokenizer.tokenize(pos_ent_b)
    tokens_neg_ent_a = bert_tokenizer.tokenize(neg_ent_a)
    tokens_neg_ent_b = bert_tokenizer.tokenize(neg_ent_b)


    pos_entity1_ids = bert_tokenizer.convert_tokens_to_ids(tokens_pos_ent_a)
    pos_entity2_ids = bert_tokenizer.convert_tokens_to_ids(tokens_pos_ent_b)
    pos_rel_id = wn_tokenizer.convert_tokens_to_ids(pos_rel)

    neg_entity1_ids = bert_tokenizer.convert_tokens_to_ids(tokens_neg_ent_a)
    neg_entity2_ids = bert_tokenizer.convert_tokens_to_ids(tokens_neg_ent_b)
    neg_rel_id = wn_tokenizer.convert_tokens_to_ids(neg_rel)


    pos_entity1_mask = [1] * len(pos_entity1_ids)
    pos_entity2_mask = [1] * len(pos_entity2_ids)
    neg_entity1_mask = [1] * len(neg_entity1_ids)
    neg_entity2_mask = [1] * len(neg_entity2_ids)


    # Zero-pad up to the sequence length.
    while len(pos_entity1_ids) < max_seq_length:
        pos_entity1_ids.append(0)
        pos_entity1_mask.append(0)
    while len(pos_entity2_ids) < max_seq_length:
        pos_entity2_ids.append(0)
        pos_entity2_mask.append(0)
    while len(neg_entity1_ids) < max_seq_length:
        neg_entity1_ids.append(0)
        neg_entity1_mask.append(0)
    while len(neg_entity2_ids) < max_seq_length:
        neg_entity2_ids.append(0)
        neg_entity2_mask.append(0)


    assert len(pos_entity1_ids) == max_seq_length
    assert len(pos_entity2_ids) == max_seq_length
    assert len(neg_entity1_ids) == max_seq_length
    assert len(neg_entity2_ids) == max_seq_length
    assert len(pos_entity1_mask) == max_seq_length
    assert len(pos_entity2_mask) == max_seq_length
    assert len(neg_entity1_mask) == max_seq_length
    assert len(neg_entity2_mask) == max_seq_length
    assert len(pos_rel_id) == 1
    assert len(neg_rel_id) == 1

    feature = InputFeaturesIdentifiablePaired(
        pos_ent1_ids=pos_entity1_ids,
        pos_ent2_ids=pos_entity2_ids,
        neg_ent1_ids=neg_entity1_ids,
        neg_ent2_ids=neg_entity2_ids,
        pos_ent1_mask=pos_entity1_mask,
        pos_ent2_mask=pos_entity2_mask,
        neg_ent1_mask=neg_entity1_mask,
        neg_ent2_mask=neg_entity2_mask,
        pos_rel_id=pos_rel_id,
        neg_rel_id=neg_rel_id)
    return feature


def convert_single_example_identifiable_paired_word_level(ex_index, example, label_list, max_seq_length, bert_tokenizer, wn_tokenizer):
    # TODO: Continue here
    triple_pos = example.triple_pos.split("; ")
    triple_neg = example.triple_neg.split("; ")
    assert len(triple_pos) == 3
    assert len(triple_neg) == 3
    pos_ent_a, pos_ent_b, pos_rel = triple_pos
    neg_ent_a, neg_ent_b, neg_rel = triple_neg
    pos_rel = [pos_rel]
    neg_rel = [neg_rel]

    tokens_pos_ent_a = pos_ent_a.split('_')
    tokens_pos_ent_b = pos_ent_b.split('_')
    tokens_neg_ent_a = neg_ent_a.split('_')
    tokens_neg_ent_b = neg_ent_b.split('_')

    pos_entity1_ids = bert_tokenizer.convert_tokens_to_ids(tokens_pos_ent_a)
    pos_entity2_ids = bert_tokenizer.convert_tokens_to_ids(tokens_pos_ent_b)
    pos_rel_id = wn_tokenizer.convert_tokens_to_ids(pos_rel)

    neg_entity1_ids = bert_tokenizer.convert_tokens_to_ids(tokens_neg_ent_a)
    neg_entity2_ids = bert_tokenizer.convert_tokens_to_ids(tokens_neg_ent_b)
    neg_rel_id = wn_tokenizer.convert_tokens_to_ids(neg_rel)

    if len(pos_entity1_ids) == 0 or len(pos_entity2_ids) == 0 or len(neg_entity1_ids) == 0 or len(neg_entity2_ids) == 0:
        return None
    pos_entity1_mask = [1] * len(pos_entity1_ids)
    pos_entity2_mask = [1] * len(pos_entity2_ids)
    neg_entity1_mask = [1] * len(neg_entity1_ids)
    neg_entity2_mask = [1] * len(neg_entity2_ids)

    # Zero-pad up to the sequence length.
    while len(pos_entity1_ids) < max_seq_length:
        pos_entity1_ids.append(0)
        pos_entity1_mask.append(0)
    while len(pos_entity2_ids) < max_seq_length:
        pos_entity2_ids.append(0)
        pos_entity2_mask.append(0)
    while len(neg_entity1_ids) < max_seq_length:
        neg_entity1_ids.append(0)
        neg_entity1_mask.append(0)
    while len(neg_entity2_ids) < max_seq_length:
        neg_entity2_ids.append(0)
        neg_entity2_mask.append(0)

    assert len(pos_entity1_ids) == max_seq_length
    assert len(pos_entity2_ids) == max_seq_length
    assert len(neg_entity1_ids) == max_seq_length
    assert len(neg_entity2_ids) == max_seq_length
    assert len(pos_entity1_mask) == max_seq_length
    assert len(pos_entity2_mask) == max_seq_length
    assert len(neg_entity1_mask) == max_seq_length
    assert len(neg_entity2_mask) == max_seq_length
    assert len(pos_rel_id) == 1
    assert len(neg_rel_id) == 1

    if ex_index < 5:
        print("*** Example ***")
        print("guid: %s" % (ex_index))
        print("tokens pos ent_a: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_pos_ent_a]))
        print("tokens pos ent_b: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_pos_ent_b]))
        print("input_ids pos ent_a: %s" % " ".join([str(x) for x in pos_entity1_ids]))
        print("input_ids pos ent_b: %s" % " ".join([str(x) for x in pos_entity2_ids]))
        print("input_mask pos ent_a: %s" % " ".join([str(x) for x in pos_entity1_mask]))
        print("input_mask pos ent_b: %s" % " ".join([str(x) for x in pos_entity2_mask]))

    feature = InputFeaturesIdentifiablePaired(
        pos_ent1_ids=pos_entity1_ids,
        pos_ent2_ids=pos_entity2_ids,
        neg_ent1_ids=neg_entity1_ids,
        neg_ent2_ids=neg_entity2_ids,
        pos_ent1_mask=pos_entity1_mask,
        pos_ent2_mask=pos_entity2_mask,
        neg_ent1_mask=neg_entity1_mask,
        neg_ent2_mask=neg_entity2_mask,
        pos_rel_id=pos_rel_id,
        neg_rel_id=neg_rel_id)
    return feature


def convert_single_example_paired(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`.
    >>> wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(); \
    all_triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples[:100]); \
    triples_explicit = WordNetProcessor()._join_wn_information_paired(wn_terms, wn_rels, all_triples); \
    examples = WordNetProcessor()._create_examples_paired(triples_explicit, "TRAIN"); \
    feature = convert_single_example_paired(1, examples[0], 50, tokenization.FullTokenizer(vocab_file='./../data/vocab_extended.txt', do_lower_case=True)); \
    """

    tokens_pos = tokenizer.tokenize(example.triple_pos)
    tokens_neg = tokenizer.tokenize(example.triple_neg)

    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], triple_pos, [SEP], triple_neg, [SEP] with "- 3"
    _truncate_seq_pair(tokens_pos, tokens_neg, max_seq_length - 3)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    # this is our first entity
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_pos:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    # this is the second entity
    for token in tokens_neg:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = 0
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


'''
copied from run_classifier.py such that the function uses the right functions
'''
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, bert_tokenizer, output_file, paired=False, multiclass=False, relation_tokenizer=None, extended_segments=False):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        if not paired and not multiclass:
            feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, bert_tokenizer)
        elif paired:
            feature = convert_single_example_paired(ex_index, example,
                                             max_seq_length, bert_tokenizer)
        elif multiclass:
            feature = convert_single_example_relation_classification(
                ex_index=ex_index,
                example=example,
                max_seq_length=max_seq_length,
                bert_tokenizer=bert_tokenizer,
                rel_tokenizer=relation_tokenizer,
                label_list=None,
                extended_segments=extended_segments
            )

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if not multiclass:
            features["label_ids"] = create_int_feature([feature.label_id])
        else:
            features["label_ids"] = create_int_feature(feature.label_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


'''
Convert e1, e2, label to features (for ivans constraints)
'''
def file_based_convert_pair_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_pair(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())




def file_based_convert_examples_to_features_identifiable(
        examples=[], label_list=[], max_seq_length=0, bert_tokenizer=None, wn_tokenizer=None, output_file='', label_entities_tokenizer=None):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        if wn_tokenizer:
            feature = convert_single_example_identifiable(ex_index=ex_index, example=example, label_list=label_list, max_seq_length=max_seq_length, bert_tokenizer=bert_tokenizer, wn_tokenizer=wn_tokenizer, label_entities_tokenizer=label_entities_tokenizer)
        else:
            feature = convert_single_example_identifiable(ex_index=ex_index, example=example, label_list=label_list,
                                                          max_seq_length=max_seq_length, bert_tokenizer=bert_tokenizer,
                                                          wn_tokenizer=None, label_entities_tokenizer=None)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()

        features["ent1_ids"] = create_int_feature(feature.ent1_ids)
        features["ent1_mask"] = create_int_feature(feature.ent1_mask)
        features["ent2_ids"] = create_int_feature(feature.ent2_ids)
        features["ent2_mask"] = create_int_feature(feature.ent2_mask)
        features["rel_id"] = create_int_feature(feature.rel_id)
        features["label_id"] = create_int_feature([feature.label_id])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_convert_examples_to_features_identifiable_paired(
    examples, max_seq_length, bert_tokenizer, wn_tokenizer, output_file, word_level=True):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)

    oov_examples = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        if word_level == True:
            feature = convert_single_example_identifiable_paired_word_level(ex_index, example, None, max_seq_length, bert_tokenizer, wn_tokenizer)
            if not feature:
                print("OOV example %d, %s" % (ex_index, example.to_string()))
                oov_examples += 1
                continue
        else:
            feature = convert_single_example_identifiable_paired(ex_index, example, None, max_seq_length, bert_tokenizer, wn_tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()

        features["pos_ent1_ids"] = create_int_feature(feature.pos_ent1_ids)
        features["pos_ent1_mask"] = create_int_feature(feature.pos_ent1_mask)
        features["pos_ent2_ids"] = create_int_feature(feature.pos_ent2_ids)
        features["pos_ent2_mask"] = create_int_feature(feature.pos_ent2_mask)
        features["pos_rel_id"] = create_int_feature(feature.pos_rel_id)
        features["neg_ent1_ids"] = create_int_feature(feature.neg_ent1_ids)
        features["neg_ent1_mask"] = create_int_feature(feature.neg_ent1_mask)
        features["neg_ent2_ids"] = create_int_feature(feature.neg_ent2_ids)
        features["neg_ent2_mask"] = create_int_feature(feature.neg_ent2_mask)
        features["neg_rel_id"] = create_int_feature(feature.neg_rel_id)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

    print("Number of oov examples %d" % (oov_examples))



def write_input_data(vocab_file_bert='./../vocab_extended.txt', do_lower_case=False, num_negatives=1000, max_sequence_length=128, output_file='./../wordnet_small.tfrecords', multiclass=False, vocab_file_rel=None, extended_segments=False, load_from_disk=False):
    '''
    >>> write_input_data()
    '''
    wn_terms, wn_rels, triples = WordNetProcessor()._load_wn(filter_vocab=False)
    print("Number of positive triples: ", str(len(triples)))

    if not multiclass:
        if load_from_disk:
            all_triples = pickle.load(open("./../data/wn_paired.p", "rb"))
            triples_pos = []
            triples_neg = []
            for pos_triple, neg_triple in all_triples:
                triples_pos.append(pos_triple)
                triples_neg.append(neg_triple)
        else:
            triples_pos, triples_neg = WordNetProcessor()._create_negative_examples(wn_terms, wn_rels, triples, len(triples))
        print("Number of negative triples: ", str(len(triples_neg)))
        ## todo: remove this! this is only for testing purposes:
        ##triples_pos = triples_pos[:1000]
        triples = triples_pos + triples_neg
        print("Total number of triples: ", str(len(triples)))
        random.shuffle(triples)
        print("Shuffled triples")

    triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, triples)
    examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN")
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_bert, do_lower_case=do_lower_case)
    if multiclass:
        relation_tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_rel,do_lower_case=False)
        file_based_convert_examples_to_features(examples=examples, label_list=None, bert_tokenizer=bert_tokenizer, max_seq_length=max_sequence_length,
                                                multiclass=multiclass, paired=False, relation_tokenizer=relation_tokenizer, output_file=output_file, extended_segments=extended_segments)
    else:
        file_based_convert_examples_to_features(examples, WordNetProcessor().get_labels(), max_sequence_length, bert_tokenizer, output_file)
    print("Done")



def write_input_data_paired(vocab_file='./../data/vocab_extended.txt', do_lower_case=True,
                     max_sequence_length=128, output_file='./../wordnet_paired_bert.tfrecords'):
    '''
    >>> write_input_data_paired()
    '''
    wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn()
    #print("Number of positive triples: ", str(len(wn_triples)))

    #all_triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples)
    #random.shuffle(all_triples)
    #print("Dump all the triples")
    #pickle.dump(all_triples, open("./../data/wn_paired.p", "wb"))

    all_triples = pickle.load(open("./../data/wn_paired.p", "rb"))
    triples_explicit = WordNetProcessor()._join_wn_information_paired(wn_terms, wn_rels, all_triples)
    examples = WordNetProcessor()._create_examples_paired(triples_explicit, "TRAIN")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    print("Tokenized triples")
    file_based_convert_examples_to_features(examples, WordNetProcessor().get_labels(), max_sequence_length, tokenizer,
                                            output_file, True)
    print("Done")


def write_input_data_identifiable_paired(bert_vocab_file='./../data/vocab.txt', wn_vocab_file='./../data/vocab_relations.txt', do_lower_case=True,
                            max_sequence_length=128, output_file='./../wordnet_paired_identifiable.tfrecords', word_level=True):
    '''
    >>> write_input_data_paired()
    '''
    wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(filter_vocab=True, vocab=tokenization.load_vocab(bert_vocab_file))
    print("Number of positive triples: ", str(len(wn_triples)))

    all_triples = WordNetProcessor()._create_negative_examples_paired(wn_terms, wn_rels, wn_triples)
    random.shuffle(all_triples)
    print("Dump all the triples")
    pickle.dump(all_triples, open("./../data/wn_paired_filtered.p", "wb"))

    all_triples = pickle.load(open("./../data/wn_paired_filtered.p", "rb"))
    triples_explicit = WordNetProcessor()._join_wn_information_paired(wn_terms, wn_rels, all_triples)
    examples = WordNetProcessor()._create_examples_paired(triples_explicit, "TRAIN")
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=do_lower_case)
    wn_tokenizer = tokenization.FullTokenizer(vocab_file=wn_vocab_file, do_lower_case=do_lower_case)

    print("Loaded triples")
    file_based_convert_examples_to_features_identifiable_paired(examples, max_sequence_length, bert_tokenizer, wn_tokenizer,
                                            output_file, word_level=word_level)
    print("Done")


def write_input_data_identifiable(bert_vocab_file='./../data/vocab.txt',
                                  wn_vocab_file='./../data/vocab_relations.txt', do_lower_case=True,
                                  max_sequence_length=128,
                                  output_file='./../data/wordnet_identifiable.tfrecords', word_level=False, entities_vocab=None):
    '''
    >>> write_input_data_identifiable()
    '''
    wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(filter_vocab=False)
                                                                #vocab=tokenization.load_vocab(bert_vocab_file))
    print("Number of positive triples: ", str(len(wn_triples)))
    triples_explicit = WordNetProcessor()._join_wn_information(wn_terms, wn_rels, wn_triples)
    examples = WordNetProcessor()._create_examples(triples_explicit, "TRAIN")
    bert_tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file, do_lower_case=do_lower_case)
    wn_tokenizer = tokenization.FullTokenizer(vocab_file=wn_vocab_file, do_lower_case=do_lower_case)

    if entities_vocab:
        entities_tokenizer = tokenization.FullTokenizer(vocab_file=entities_vocab, do_lower_case=False)

    print("Loaded triples")
    if entities_tokenizer:
        file_based_convert_examples_to_features_identifiable(examples=examples, label_list=[0],
                                                             max_seq_length=max_sequence_length,
                                                             bert_tokenizer=bert_tokenizer,
                                                             wn_tokenizer=wn_tokenizer,
                                                             output_file=output_file, label_entities_tokenizer=entities_tokenizer)
    else:
        file_based_convert_examples_to_features_identifiable(examples=examples, label_list=[0], max_seq_length=max_sequence_length, bert_tokenizer=bert_tokenizer,
                                                                wn_tokenizer=wn_tokenizer,
                                                                output_file=output_file)
    print("Done")

'''
Copied from run_classifier.py
'''
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_entities_matrix(vocab_file='./../data/vocab.txt', do_lower_case=True):
    wn_terms, wn_rels, wn_triples = WordNetProcessor()._load_wn(filter_vocab=False)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    wn_ids = []
    wn_masks = []
    wn_tokens = []
    wn_terms_list = list(wn_terms.values())
    for term in wn_terms_list:
        tokens = tokenizer.tokenize(term)
        wn_tokens.append(tokens)
        wn_masks.append([1 for i in range(len(tokens))])
        wn_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    print(len(wn_ids))
    max_length = max([len(wn_id_list) for wn_id_list in wn_ids])
    for i,wn_id_list in enumerate(wn_ids):
        while len(wn_id_list) < max_length:
            wn_id_list.append(0)
            wn_masks[i].append(0)
    with codecs.open("./../data/wn_terms_vocab.txt", "w", 'utf8') as f:
        for term in wn_terms_list:
            f.write(term)
            f.write("\n")
        f.close()
    pickle.dump(wn_tokens, open("./../data/wn_tokens.p", "wb"))
    pickle.dump(wn_ids, open("./../data/wn_wordpiece_ids.p", "wb"))
    pickle.dump(wn_masks, open("./../data/wn_masks.p", "wb"))



def clean_constraint(token):
    token = token.replace("en_", "", 1)
    token = token.replace("_", " ").lower().strip()
    return token


def load_syn_hyp_constraints(path="./../data/syn_hyp1_constraints_ivan.txt"):
    constraints = []
    with codecs.open(path, "r", "utf8") as f:
        for line in f.readlines():
            word_1, word_2 = line.split(" ")
            word_1 = clean_constraint(word_1)
            word_2 = clean_constraint(word_2)
            constraints.append([word_1, word_2, True])
    return constraints


def mat_normalize(mat, norm_order=2, axis=1):
    try:
        return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])
    except Exception as e:
        print(e)


def cosine(a, b):
    norm_a = mat_normalize(a)
    norm_b = mat_normalize(b)
    cos = np.dot(norm_a, np.transpose(norm_b))
    return cos


def generate_embd_matrix(batch_vocab, embd_dict):
    # and compute pairwise distances
    batch_embd_matrix = []
    batch_vocab_term_to_index = {}
    batch_vocab_index_to_term = {}
    for i, term in enumerate(batch_vocab):
        term_vectors = []
        for part in term.split(" "):
            if part in embd_dict:
                term_vectors.append(embd_dict[part])
            elif part.capitalize() in embd_dict:
                term_vectors.append(embd_dict[part.capitalize()])
            else:
                print("%s not in embedding dict" % part)
        if len(term_vectors) > 1:
            term_vector = np.average(term_vectors, axis=0)
        elif len(term_vectors) == 1:
            term_vector = term_vectors[0]
        else:
            term_vector = [0 for i in range(300)]
        batch_embd_matrix.append(np.array(term_vector))
        batch_vocab_term_to_index[term] = i
        batch_vocab_index_to_term[i] = term
    return batch_vocab_term_to_index, batch_vocab_index_to_term, np.array(batch_embd_matrix)



def get_batch_closest_neighbor(term, forbidden_term, true_constraints, similarity_matrix, batch_vocab_term_to_index, batch_vocab_index_to_term, true_constraints_dict):
    # get batch-closest neighbor to w1
    term_index = batch_vocab_term_to_index[term]
    term_similarities = similarity_matrix[term_index]
    term_similarities_indices = term_similarities.argsort()[::-1]
    for index in term_similarities_indices:
        # Term should not be nan vector; should not be the same term and not be the correct partner ter,.
        if not math.isnan(term_similarities[index]) and index != term_index and index != batch_vocab_term_to_index[forbidden_term] and term + "_" + batch_vocab_index_to_term[index] not in true_constraints_dict and batch_vocab_index_to_term[index] + "_" + term not in true_constraints_dict: #[term, batch_vocab_index_to_term[index], True] not in true_constraints_dict:
            # this is the index of the new w1
            return batch_vocab_index_to_term[index]
    return None


def convert_contraints_list_to_dict(true_constraints_list):
    true_constraints_dict = {}
    for t1,t2,truth in true_constraints_list:
        true_constraints_dict[t1 + "_" + t2] = truth
    return true_constraints_dict


def smart_negative_sampling(true_constraints, true_constraints_dict):
    embd_dict = utils.load_embeddings("./../data/fasttext/wiki-news-300d-1M-subword.vec")
    #print("Embedding test")
    #print("aachen" in embd_dict)
    #print("Aachen" in embd_dict)
    all_examples = []
    for i,batch in enumerate(batch_helper.batch_iter(true_constraints, 32, 1, False)):
        if i % 1000 == 0:
            print("Batch %d" % i)
        batch_negative_examples = []
        batch_vocab = []
        for t1,t2,truth in batch:
            batch_vocab.append(t1)
            batch_vocab.append(t2)
        batch_vocab = list(set(batch_vocab))
        batch_vocab_term_to_index, batch_vocab_index_to_term, batch_embd_matrix = generate_embd_matrix(batch_vocab, embd_dict)
        similarity_matrix = cosine(batch_embd_matrix, batch_embd_matrix)
        for i, (w1, w2, truth) in enumerate(batch):
            # get batch-closest neighbor to w1
            wx = get_batch_closest_neighbor(w1, w2, true_constraints, similarity_matrix, batch_vocab_term_to_index, batch_vocab_index_to_term, true_constraints_dict)
            wy = get_batch_closest_neighbor(w2, w1, true_constraints, similarity_matrix, batch_vocab_term_to_index, batch_vocab_index_to_term, true_constraints_dict)
            all_examples.append([w1, w2, True])
            if wx is not None:
                batch_negative_examples.append([w1, wx, False])
                all_examples.append([w1, wx, False])
            if wy is not None:
                batch_negative_examples.append([w2, wy, False])
                all_examples.append([w2, wy, False])
    return all_examples


def create_data_syn_hyp_constraints():
    '''
    >>> create_data_syn_hyp_constraints()
    '''
    # load constraints
    true_constraints = load_syn_hyp_constraints()
    # remove duplicates (due to lowercasing we might have some)
    #true_constraints = np.array(set(true_constraints))
    true_constraints.sort()
    true_constraints = list(k for k, _ in itertools.groupby(true_constraints))
    true_constraints_dict = convert_contraints_list_to_dict(true_constraints)

    # TODO: only for testing! Remove this!
    #true_constraints = true_constraints[:1000]
    ###########

    random.shuffle(true_constraints)

    print("Number of positive examples %d" % len(true_constraints))
    all_constraints = smart_negative_sampling(true_constraints, true_constraints_dict)
    print("Number of all examples %d" % len(all_constraints))
    pickle.dump(all_constraints, open("./../data/all_constraints.p", "wb"))



def write_input_data_syn_hyp_constraints(vocab_file="./../data/BERT_base_new/vocab.txt", output_file="./../data/syn_hyp1_constraints_ivan_2_test.tfrecord"):
    all_contraints = pickle.load(open("./../data/all_constraints.p", "rb"))
    all_contraints = create_input_pair(all_contraints, "TRAIN")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_pair_to_features(examples=all_contraints, label_list=[True, False],
                                                         max_seq_length=128,
                                                         tokenizer=tokenizer,
                                                         output_file=output_file)


def main():
    #create_data_syn_hyp_constraints()
    write_input_data_syn_hyp_constraints()

if __name__=="__main__":
    main()
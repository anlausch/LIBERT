import tensorflow as tf
from util.debug_util import f


vocabulary_indices = tf.Variable([1, 2, 3])
vocab_size=5
def sentence2tfidf(vocabulary_indices, vocab_size):
  """
  Takes an array of vocabulary indices and turns it into a tf-idf representation. E.g. for a sentence
  represented by its vocabulary token ids [0, 1, 2] is turned into a tf-idf sentence matrix [[0,0,1],[0,1,0],[0,0,1]].
  Use tf.map_fn to apply this to a batch of sentences in order to obtain a tfidf tensor.
  :param vocabulary_indices: a list of word indices in vocabulary
  :param vocab_size: determines the length of tfidf vectors, i.e. the max value that can appear in vocabulary_indices
  :return:
  """
  seq_len=vocabulary_indices.shape[0]
  vocab_indices_expanded = tf.expand_dims(vocabulary_indices, -1)

  ones = tf.reshape(tf.ones_like(vocab_indices_expanded), [-1]) # a.k.a. tfidf values
  column_indices = tf.range(vocabulary_indices.shape[0])
  column_indices_expanded = tf.expand_dims(column_indices, -1)

  tfidf_indices = tf.concat([vocab_indices_expanded, column_indices_expanded], 1)
  tfidf_indices_int64 = tf.cast(tfidf_indices,dtype=tf.int64)
  sparse = tf.SparseTensor(indices=tfidf_indices_int64, values=ones, dense_shape=[vocab_size, seq_len])
  return sparse
# dense = tf.sparse_tensor_to_dense(sparse)
# test=tf.SparseTensor(indices=[(0,0,0),(0,1,1)], values=[1,2], dense_shape=[2,2,1])


# import tensorflow as tf
# import numpy as np
#
# vocabulary_size = 10
# embedding_size = 1
#
#
# lookupindices = tf.SparseTensor(indices=list(range(8)), values=[1]*8, dense_shape=[8])
# dense = tf.sparse_tensor_to_dense(lookupindices)
# # test = tf.get_variable(initializer=[3, 3, 4], dtype=tf.int32, name="test")
#
# # embeddings = tf.Variable(np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0,
# #                                    49.0, 64.0, 81.0]))
# embeddings = tf.Variable(np.array([[0.0, 1.0],
#                                    [4.0, 9.0],
#                                    [16.0, 25.0],
#                                    [36.0,49.0],
#                                    [64.0, 81.0]]))
# # embeddings = tf.Variable(np.array([]))
#
# embed = tf.nn.embedding_lookup_sparse(embeddings, lookupindices, None)
#
# with tf.Session() as s:
#   s.run(tf.initialize_all_variables())
#   print(s.run(dense))
#   print(s.run(embed))

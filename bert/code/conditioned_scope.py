import tensorflow as tf
"""
Extension for RESCAL multitask etc..
"""
class empty_scope():
  def __init__(self):
    pass

  def __enter__(self):
    pass

  def __exit__(self, type, value, traceback):
    pass


def cond_scope(is_shared=False):
  return empty_scope()
  #return empty_scope() if not is_shared else tf.variable_scope("", reuse=tf.AUTO_REUSE)
  #return empty_scope() if not is_shared else tf.variable_scope("shared", reuse=tf.AUTO_REUSE)
  #return empty_scope() if not is_shared else tf.variable_scope("shared_embedding_matrix", reuse=tf.AUTO_REUSE)
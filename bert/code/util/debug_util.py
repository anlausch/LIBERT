import tensorflow as tf

def f(fetches, checkpoint_dir=None):
  if checkpoint_dir is None:
    init = tf.global_variables_initializer()
  else:
    init = tf.train.init_from_checkpoint()
  tmp=tf.Session()
  tmp.run(init)
  return tmp.run(fetches)

tmpsess = None

def g(fetches, newsess=False, checkpoint_dir=None):
  global tmpsess
  if tmpsess is None or newsess:
    tmpsess = tf.Session()

  if checkpoint_dir is None:
    init = tf.global_variables_initializer()
    tmpsess.run(init)
  else:
    saver = tf.train.Saver()
    ckpt =tf.train.latest_checkpoint(
      checkpoint_dir,
      latest_filename=None
    )
    saver.restore(tmpsess, ckpt)

  return tmpsess.run(fetches)

# def h(path):
#   example = next(tf.python_io.tf_record_iterator(path))
#   return example

import tensorflow as tf

def f(fetches):
  init = tf.global_variables_initializer()
  tmp=tf.Session()
  tmp.run(init)
  return tmp.run(fetches)
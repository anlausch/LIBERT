import tensorflow as tf
from tensorflow.python import debug as tf_debug

def my_op(x, scalar_name):
  var1 = tf.get_variable(scalar_name,
                         shape=[],
                         initializer=tf.constant_initializer(10.0, dtype=tf.float32))
  return x * var1

scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')

z = scale_by_y(tf.constant(2.0, dtype=tf.float32))
w = scale_by_y(tf.constant(5.0, dtype=tf.float32))

s = tf.Session()
s = tf_debug.TensorBoardDebugWrapperSession(s, "localhost:6064")
s.run([z,w])
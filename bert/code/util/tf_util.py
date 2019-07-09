import tensorflow as tf

def get_variable_by_name(name, collection):
  variable = [v for v in tf.get_collection(collection) if v.name == name]
  if len(variable) == 0:
    raise NameError("Variable %s not found in collection %s" % (name, collection))
  return variable[0]

def get_metric_ops_by_name(name):
  ops = [op for op in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) if name in op.name]
  if len(ops) == 0:
    raise NameError("Metric %s not found in %s" % (name, tf.GraphKeys.METRIC_VARIABLES))
  elif len(ops) != 2:
    raise AssertionError("Metric %s does not contain exactly two ops, ensure each metric is uniquely named." % name)

  if "total" in ops[0]:
    total_op = ops[0]
    count_op = ops[1]
  else:
    total_op = ops[1]
    count_op = ops[0]

  return total_op, count_op

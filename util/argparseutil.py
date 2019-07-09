import argparse
import re

# from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

# from https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
class StoreModelParams(argparse.Action):
  def __init__(self, option_strings, dest, nargs=None, **kwargs):
    self._nargs = nargs
    super(StoreModelParams, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    model_params = {}
    float_pattern = '^[0-9]+\.{1}[0-9]+$'
    int_pattern = '^[0-9]+$'
    for kv in values:
      k, v = kv.split("=")
      if re.search(float_pattern, v):
        v=float(v)
      elif re.search(int_pattern, v):
        v=int(v)
      elif v.lower() == "true":
        v=True
      elif v.lower() == "false":
        v=False
      model_params[k] = v
    setattr(namespace, self.dest, model_params)

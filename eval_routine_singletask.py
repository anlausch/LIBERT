import tensorflow as tf
import importlib
import pathlib
from functools import partial
from checkmate import BestCheckpointSaver
import argparse
from util.tf_util import get_variable_by_name
from util import gputil
from util.argparseutil import StoreModelParams


# Mandatory arguments
parser = argparse.ArgumentParser(description="Running a models' evaluation routine")
parser.add_argument("--data_path", type=str, help="directory containing evaluation tfrecords", required=True)
parser.add_argument("--input_fn", type=str, help="any of {get_nmt_data, get_nli_data}")
parser.add_argument("--model_dir", help="directory containing checkpoints", required=True)
parser.add_argument("--model_fn", type=str, help="One of the function names in model.model_functions",
                    required=True)

# Optional arguments
parser.add_argument("--checkpoint_dir", type=str, help="Refer to Estimator.evaluate checkpoint_dir argument",
                    default=None)
parser.add_argument("--eval_name", type=str, help="Suffixes eval folder name inside MODEL_HOME, "
                                                  "e.g. \"dev\" leads to results written in MODEL_HOME/eval_dev")
parser.add_argument("-d", "--cuda_visible_devices", type=str, help="visible GPUs")
parser.add_argument("-f", "--gpu_fraction", type=float, help="percentage of GPU memory to occupy.")
parser.add_argument("-p", "--params", action=StoreModelParams, nargs="+", dest="params")
args = parser.parse_args()
gpu_options = gputil.set_gpus_from_args(args)


class BestCheckpointSaverHook(tf.train.SessionRunHook):
  def __init__(self, path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    self._saver = None
    self.path = path

  def begin(self):
    self._saver = BestCheckpointSaver(
      save_dir=self.path,
      num_to_keep=3,
      maximize=False
    )

  def end(self, session):
    # TODO: create sep. file for outsourcing var name constants
    count_var = get_variable_by_name('mean/count:0', tf.GraphKeys.METRIC_VARIABLES)
    sum_var = get_variable_by_name('mean/total:0', tf.GraphKeys.METRIC_VARIABLES)

    count, _sum = session.run([count_var, sum_var])
    loss = _sum / count

    tf.logging.info("loss value is %s" % str(loss))
    global_step_tensor = tf.train.get_global_step()
    self._saver.handle(loss, session, global_step_tensor)


def get_model_fn():
  modules = ['model.argrec']
  _module = importlib.import_module('model.argrec')
  return getattr(_module, args.model_fn)

def get_data_fn():
  _module = importlib.import_module('replantio.get_data')
  data_fn = getattr(_module, args.input_fn)
  return partial(data_fn, path=args.data_path, epochs=1, is_train=False, cut_final_batch=True)

def main():
  model_dir = args.model_dir
  is_part_of_multitask_eval = args.eval_name == "tmp"
  model_fn = get_model_fn()
  get_eval_data_fn = get_data_fn()

  hooks = None
  if not is_part_of_multitask_eval: # "tmp" indicates that this run is part of multitask eval
    best_model_dir = model_dir + "eval/"
    best_model_dir += "best_model/"
    hooks = [BestCheckpointSaverHook(best_model_dir)]

  runconfig = tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=gpu_options))
  estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=args.params, config=runconfig)
  r = estimator.evaluate(get_eval_data_fn, hooks=hooks, name=args.eval_name, checkpoint_path=args.checkpoint_dir)
  print(r, end="")


if __name__ == '__main__':
  main()

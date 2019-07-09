import tensorflow as tf
import importlib
import pathlib
from functools import partial
from checkmate import BestCheckpointSaver
import argparse
from util.tf_util import get_variable_by_name
from util import gputil
import modeling
from util.argparseutil import StoreModelParams


# Mandatory arguments
parser = argparse.ArgumentParser(description="Running a models' evaluation routine")
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
parser.add_argument("-p", "--params", action=StoreModelParams, nargs="*", dest="params")

# all this is needed because we are working with function builders and not with actual named functions
parser.add_argument("--input_file", type=str, help="Evaluation files for the standard tasks", required=True)
parser.add_argument("--bert_config_file", type=str, help="Config file for bert encoder", required=True)
parser.add_argument("--eval_batch_size", type=int, help="Evaluation batch size", required=True)
parser.add_argument("--max_sequence_length", type=int, help="Config file for bert encoder", required=True)
parser.add_argument("--max_predictions_per_sequence", type=int, help="Max predictions per sequence Evaluation batch size", required=True)

parser.add_argument("--max_eval_steps", type=int, default=100, help="Maximum number of eval steps.")

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
  _module = importlib.import_module('run_pretraining_wordnet')
  return getattr(_module, args.model_fn)

def get_data_fn():
  _module = importlib.import_module('run_pretraining_wordnet')
  data_fn = getattr(_module, args.input_fn)
  return data_fn

def main():
  model_dir = args.model_dir
  is_part_of_multitask_eval = args.eval_name == "tmp"
  model_fn_builder = get_model_fn()
  get_eval_data_fn_builder = get_data_fn()

  input_files = []
  for input_pattern in args.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

  input_fn = get_eval_data_fn_builder(
    input_files=input_files,
    max_seq_length=args.max_sequence_length,
    max_predictions_per_seq=args.max_predictions_per_sequence,
    is_training=False)

  model_fn = model_fn_builder(bert_config=bert_config,
                              init_checkpoint=args.checkpoint_dir,
                              learning_rate=0.0,
                              num_train_steps=0,
                              num_warmup_steps=0,
                              use_one_hot_embeddings=True,
                              use_tpu=False)

  hooks = None
  if not is_part_of_multitask_eval: # "tmp" indicates that this run is part of multitask eval
    best_model_dir = model_dir + "eval/"
    best_model_dir += "best_model/"
    hooks = [BestCheckpointSaverHook(best_model_dir)]

  #runconfig = tf.estimator.RunConfig(session_config=tf.ConfigProto(gpu_options=gpu_options))

  runconfig = tf.contrib.tpu.RunConfig(
      model_dir=model_dir
  )
  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=runconfig,
      eval_batch_size=args.eval_batch_size,
      train_batch_size=args.eval_batch_size)
  #estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir, params=args.params, config=runconfig)
  r = estimator.evaluate(input_fn=input_fn, hooks=hooks, name=args.eval_name, checkpoint_path=args.checkpoint_dir, steps=args.max_eval_steps)
  print(r, end="")


if __name__ == '__main__':
  main()

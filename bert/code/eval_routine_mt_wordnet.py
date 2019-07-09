import os
import tensorflow as tf
import time
import subprocess
import wn_config as c
import modeling
from functools import partial
from util.timer import Timer
from util.process_util import run_ssh_or_shell_command

from run_pretraining_wordnet import model_fn_builder_bert_wordnet
from run_pretraining_wordnet import model_fn_builder_bert_standard

from run_pretraining_wordnet import input_fn_builder_wordnet
from run_pretraining_wordnet import input_fn_builder_standard

from checkmate import BestCheckpointSaver
from util.gputil import disable_all_gpus
import argparse

disable_all_gpus()

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_dir", type=str, help="Name of multitask model dir",
                    required=True)
parser.add_argument("-d", "--cuda_visible_devices", default=c.DEV_CUDA_VISIBLE_DEVICES, type=str, help="visible GPUs")
parser.add_argument("-f", "--gpu_fraction", type=float, default=c.DEV_CUDA_GPU_FRAC, help=" %% of GPU memory to occupy.")
parser.add_argument("-p", "--params", nargs="*", type=str, help="estimator model parameters")


parser.add_argument("--input_file_standard", type=str, default=c.STANDARD_INPUT_DEV, help="Evaluation files for the standard tasks")
parser.add_argument("--input_file_wn", type=str, default=c.WN_INPUT_DEV, help="Evaluation files for the wn tasks")
parser.add_argument("--bert_config_file", type=str, default=c.BERT_CONFIG_FILE, help="Config file for bert encoder")
parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
parser.add_argument("--max_sequence_length", type=int, default=128, help="Config file for bert encoder")
parser.add_argument("--max_predictions_per_sequence", type=int, default=20, help="Max predictions per sequence Evaluation batch size")

parser.add_argument("--max_eval_steps", type=int, default=100, help="Maximum number of eval steps.")

args = parser.parse_args()

#PARAMS = " ".join(args.params) if args.params else None
DEV_CUDA_VISIBLE_DEVICES = args.cuda_visible_devices
DEV_CUDA_GPU_FRAC = args.gpu_fraction
MODEL_DIR = args.model_dir if args.model_dir[-1] == "/" else args.model_dir + "/"

# these are the bert function builder arguments/ tpu estimator arguments
BERT_CONFIG_FILE = args.bert_config_file
EVAL_BATCH_SIZE = args.eval_batch_size
MAX_SEQUENCE_LENGTH = args.max_sequence_length
MAX_PREDICTIONS_PER_SEQUENCE = args.max_predictions_per_sequence


def _create_summary(tag, value):
  return tf.Summary(value=[
    tf.Summary.Value(tag=tag, simple_value=value)
  ])


def _write_summaries(name_value_pairs, target_dir):
  writer = tf.summary.FileWriter(target_dir)
  global_step = name_value_pairs.pop('global_step',None)

  for name, value in name_value_pairs.items():
    writer.add_summary(_create_summary(name, value), global_step=global_step)


def _clean(directory):
  if os.path.exists(directory):
    os.remove(directory)


def _restore_graph_save(sess, attempts, wait_time_seconds):
  while attempts:
    try:
      _restore_graph(sess)
      tf.logging.info("Model restored successfully.")
      return
    except OSError:
      attempts -= 1
      wait_time_seconds *= 2
      tf.logging.warning("Couldn't load restore checkpoint file, %s more attempts before OSError, tyring again in "
                         "%s seconds" % (str(attempts), str(wait_time_seconds)))
      time.sleep(wait_time_seconds)
  _restore_graph(sess)


def _restore_graph(sess):
  """
  restores the graph without building it with model functions. has the advantage that we don't need to supply input
  feed functions (much cleaner here), since we only want to export it if its dev performance is new best dev performance
  :param sess:
  :return:
  """
  latest_ckpt = tf.train.latest_checkpoint(MODEL_DIR)
  saver = tf.train.import_meta_graph(latest_ckpt + ".meta")
  saver.restore(sess, latest_ckpt)


def _update_best_model(directory, value):
  """
  Maintains a directory in which the three best performing models are saved, measured by value.
  :param directory: where checkpoints of best models are saved.
  :param value: value by which to compare if current model performs better.
  :return:
  """
  with tf.Session() as sess:
    _restore_graph_save(sess, wait_time_seconds=90, attempts=5)
    checkpoint_saver = BestCheckpointSaver(
      save_dir=directory,
      num_to_keep=3,
      maximize=False
    )
    global_step_tensor = tf.train.get_global_step()
    checkpoint_saver.handle(value, sess, global_step_tensor)


def _launch_dummy_process():
  """
  Launches dummy process to keep gpu resources occupied, the parent process, once its done, receives
  the pid and uses it to release the resources for a new eval routine. Redirects stdout and stderr to
  dev null such that this module can be called exit without waiting for dummy subprocess to finish.
  :return:
  """
  devnull = open(os.devnull, 'w')
  proc = subprocess.Popen(["/usr/bin/nohup", c.PYTHON_RUNTIME, c.PATH_DUMMY_PROCESS_SCRIPT,
                           '--cuda_visible_devices', DEV_CUDA_VISIBLE_DEVICES,
                           "--gpu_fraction", str(DEV_CUDA_GPU_FRAC)],
                          stdout=devnull,
                          stderr=subprocess.STDOUT)
  return proc.pid


def create_dev_null_symlink(src):
  if not os.path.islink(src):
    os.symlink("/dev/null", src)


def run_single_task_eval(checkpoint_path, eval_dir_suffix, model_fn, data_fn, input_file):
  command = " ".join([c.PYTHON_RUNTIME, c.PATH_EVAL_SCRIPT,
                      '--cuda_visible_devices', DEV_CUDA_VISIBLE_DEVICES,
                      '--gpu_fraction', str(DEV_CUDA_GPU_FRAC),
                      '--model_dir', MODEL_DIR,
                      '--checkpoint_dir', checkpoint_path,
                      '--eval_name', eval_dir_suffix,
                      '--model_fn', model_fn.__name__,
                      '--input_fn', data_fn.__name__,
                      '--input_file', input_file,
                      '--bert_config_file', BERT_CONFIG_FILE,
                      '--eval_batch_size', str(EVAL_BATCH_SIZE),
                      '--max_sequence_length', str(MAX_SEQUENCE_LENGTH),
                      '--max_predictions_per_sequence', str(MAX_PREDICTIONS_PER_SEQUENCE)
                      ])
                                   #'--params', PARAMS])

  eval_process = run_ssh_or_shell_command(command)
  result, err = eval_process.communicate(timeout=20 * 30)
  result = eval(result)
  return result


def main():
  # we need to cache checkpoint here, between single task evaluations there may be a new latest checkpoint
  checkpoint = tf.train.latest_checkpoint(MODEL_DIR)
  assert checkpoint is not None
  eval_dir_suffix = "tmp"
  trash_eval_dir = MODEL_DIR + "eval_" + eval_dir_suffix
  create_dev_null_symlink(trash_eval_dir)

  # child processes will complain with
  # E tensorflow/core/util/events_writer.cc:104] Write failed because file could not be opened.
  # but that's the intended behaviour, we don't want to write anything hence the trash_eval_dir pointing
  # to /dev/null
  timer = Timer()

  evaluate_task = partial(run_single_task_eval, checkpoint_path=checkpoint, eval_dir_suffix=eval_dir_suffix)

  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Batch size = %d", EVAL_BATCH_SIZE)

  standard_result = evaluate_task(model_fn=model_fn_builder_bert_standard,
                                  data_fn=input_fn_builder_standard,
                                  input_file=args.input_file_standard
                                  )
  tf.logging.warning("Evaluated standard tasks %s (%s)" % (timer.pprint_lap(),str(standard_result)))



  wn_result = evaluate_task(model_fn=model_fn_builder_bert_wordnet, data_fn=input_fn_builder_wordnet, input_file=args.input_file_wn)
  tf.logging.warning("Evaluated wordnet task %s (%s)" % (timer.pprint_lap(),str(wn_result)))

  pid = _launch_dummy_process()
  tf.logging.error("Spawned dummy process with pid %s" % str(pid))

  global_results = {'loss': standard_result['loss'] + wn_result['loss'],
                    'multitask/standard_loss': standard_result['loss'],
                    'multitask/wn_loss': wn_result['loss'],
                    'global_step': standard_result['global_step']}

  if not (standard_result['global_step'] == wn_result['global_step']):
    raise AssertionError("Different global steps! standard=%s, wn=%s" %
                         (standard_result['global_step'], wn_result['global_step']))

  actual_eval_dir = MODEL_DIR + "eval/"
  best_model_dir = MODEL_DIR + "best_model/"

  _update_best_model(value=global_results['loss'], directory=best_model_dir)
  _write_summaries(global_results, actual_eval_dir)
  _clean(trash_eval_dir)

  global_results["pid"] = pid
  print(global_results, end="")


if __name__ == '__main__':
  main()

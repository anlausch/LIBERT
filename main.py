import argparse
import config as c
import tensorflow as tf
import logging
import os

from estimator_extension import EvalRoutineCheckpointSaverListener
#from model.treccar import multitask_model_fn
#from model.treccar import nli_model_fn
from model.treccar import multichannel_model_fn
from model.argrec import elmo_threshold_model_fn

from replantio.get_data import get_nmt_data
from replantio.get_data import get_nli_data
from replantio.get_data import get_treccar_data
from replantio.get_data import get_argrec_data
from replantio.get_data import get_argrec_data_tfrecords

from tensorflow.python import debug as tf_debug
from util.gputil import set_gpus_from_args
from util.timer import Timer
from util.process_util import run_ssh_or_shell_command
from util.argparseutil import str2bool


# Server Parameter
parser = argparse.ArgumentParser()
parser.add_argument("-train_d", "--train_cuda_visible_devices", default=c.TRAIN_CUDA_VISIBLE_DEVICES, type=str,
                    help="visible GPUs")
parser.add_argument("-train_f", "--train_gpu_fraction", type=float, default=c.TRAIN_CUDA_GPU_FRAC,
                    help="percentage of GPU memory to occupy.")
parser.add_argument("-dev_d", "--dev_cuda_visible_devices", default=c.DEV_CUDA_VISIBLE_DEVICES, type=str,
                    help="visible GPUs")
parser.add_argument("-dev_f", "--dev_gpu_fraction", type=float, default=c.DEV_CUDA_GPU_FRAC,
                    help="percentage of GPU memory to occupy.")
parser.add_argument("-dev_s", "--dev_server", default=c.DEV_SERVER, type=str)

# Model Parameter
parser.add_argument("-m","--model_dir", type=str, help="One of the function names in model.model_functions",
                    required=True)
parser.add_argument("-l", "--LAMBDA", type=float, default=1.0, help="trade-off for orthogonality loss vs task loss")
parser.add_argument("-hi", "--hidden", type=int, default=c.HIDDEN_SIZE, help="hidden size")
parser.add_argument("-lr", "--learning_rate", type=float, default=c.LEARNING_RATE)
parser.add_argument("-e", "--epochs", type=int, default=c.EPOCHS)
parser.add_argument("-te", "--tune_embeddings", type=str2bool, default=c.TUNE_EMBEDDINGS, help="boolean flag")
parser.add_argument("-sp", "--shared_private", type=str2bool, default=c.SHARED_PRIVATE, help="boolean flag")
parser.add_argument("-orth", "--impose_orthogonality", type=str2bool, default=c.IMPOSE_ORTHOGONALITY, help="boolean flag")
args = parser.parse_args()

MODEL_HOME = args.model_dir if args.model_dir[-1] == "/" else args.model_dir + "/"

EPOCHS = args.epochs
TUNE_EMBEDDINGS = args.tune_embeddings
LEARNING_RATE = args.learning_rate
SHARED_PRIVATE = args.shared_private
IMPOSE_ORTHOGONALITY = args.impose_orthogonality
LAMBDA = args.LAMBDA
HIDDEN_SIZE = args.hidden

DEV_SERVER = args.dev_server
DEV_GPU_FRACTION = args.dev_gpu_fraction
DEV_CUDA_VISIBLE_DEVICES = args.dev_cuda_visible_devices

timer = Timer()

is_new_run = False
if not os.path.exists(MODEL_HOME):
  os.makedirs(MODEL_HOME)
  is_new_run = False

FORMAT = '%(asctime)-15s %(message)s'
formatter = logging.Formatter(FORMAT)
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger("tensorflow")
logger.handlers=[]
filehandler = logging.FileHandler(MODEL_HOME + "train_eval.log")
filehandler.setLevel(logging.DEBUG)
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

if is_new_run:
  logger.info("Orthogonality: %s (lambda=%s)" % (str(IMPOSE_ORTHOGONALITY), str(LAMBDA)))
  logger.info("Shared private: %s" % str(SHARED_PRIVATE))
  logger.info("Learning rate: %s" % str(c.LEARNING_RATE))
  logger.info("")
  logger.info("Epochs: %s" % str(EPOCHS))
  logger.info("Batch size: %s" % str(c.BATCH_SIZE))
  logger.info("Hidden size: %s" % str(HIDDEN_SIZE))
  logger.info("Max Seq Len: %s" % str(c.MAX_SEQ_LEN))
  logger.info("Vocabulary size: %s" % str(c.VOCAB_SIZE))
else:
  logger.info("Continue training...")

args.cuda_visible_devices = args.train_cuda_visible_devices
args.gpu_fraction = args.train_gpu_fraction
gpu_options = set_gpus_from_args(args)


def get_tf_run_config():
  return tf.estimator.RunConfig(save_checkpoints_steps=c.SAVE_CHECKPOINT_STEPS,
                                save_checkpoints_secs=c.SAVE_CHECKPOINT_SECS,
                                model_dir=MODEL_HOME,
                                save_summary_steps=c.SAVE_SUMMARY_STEPS,
                                log_step_count_steps=c.LOG_STEP_COUNT_STEPS,
                                session_config=tf.ConfigProto(gpu_options=gpu_options))


def get_nli_run_config():
  logger.info("Launching NLI model")
  get_nli_train_data = lambda: get_nli_data(c.nli_tfrecord_train, is_train=True, epochs=EPOCHS, cut_final_batch=False)
  model_parameter = {'learning_rate': LEARNING_RATE,
                     'tune_embeddings': TUNE_EMBEDDINGS,
                     'shared_private': SHARED_PRIVATE,
                     'impose_orthogonality': IMPOSE_ORTHOGONALITY,
                     # 'pretrained_emb_en': c.PATH_FASTTEXT_EN,
                     'hidden_size': HIDDEN_SIZE}
  eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=MODEL_HOME,
                                                          path_eval_script=c.PATH_EVAL_SCRIPT,
                                                          data_path=c.nli_tfrecord_dev,
                                                          model_fn=nli_model_fn,
                                                          input_fn=get_nli_data,
                                                          gpu_fraction=DEV_GPU_FRACTION,
                                                          cuda_visible_devices=DEV_CUDA_VISIBLE_DEVICES,
                                                          server=DEV_SERVER,
                                                          params=model_parameter)
  return get_nli_train_data, nli_model_fn, model_parameter, eval_hook_listener


def get_argrec_run_config():
  logger.info("Launching ArgRec model")
  get_argrec_train_data = lambda: get_argrec_data_tfrecords(c.PROJECT_HOME + "/data/debater_tfrecords_TRAIN", is_train=True, epochs=-1, cut_final_batch=False)
  #get_argrec_train_data = lambda: get_argrec_data("TRAIN", is_train=True, epochs=-1, cut_final_batch=False)

  #get_argrec_test_data = lambda: get_argrec_data("TEST", is_train=False, epochs=1, cut_final_batch=False)
  get_argrec_test_data = lambda: get_argrec_data_tfrecords(c.PROJECT_HOME + "/data/debater_tfrecords_TEST", is_train=False, epochs=1,
                                    cut_final_batch=False)
  model_parameter = {'learning_rate': 0.0001}

  # # TODO: Call this again for testing (but with other path/ parameter .. whatever)
  # eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=MODEL_HOME,
  #                                                         path_eval_script=c.PATH_EVAL_SCRIPT,
  #                                                         data_path="DEV",
  #                                                         model_fn=elmo_threshold_model_fn,
  #                                                         input_fn=get_argrec_data,
  #                                                         gpu_fraction=c.DEV_CUDA_GPU_FRAC,
  #                                                         cuda_visible_devices=c.DEV_CUDA_VISIBLE_DEVICES,
  #                                                         server=c.DEV_SERVER,
  #                                                         params=model_parameter)
  eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=MODEL_HOME,
                                                          path_eval_script=c.PATH_EVAL_SCRIPT,
                                                          data_path=c.PROJECT_HOME + "data/debater_tfrecords_DEV",
                                                          model_fn=elmo_threshold_model_fn,
                                                          input_fn=get_argrec_data_tfrecords,
                                                          gpu_fraction=c.DEV_CUDA_GPU_FRAC,
                                                          cuda_visible_devices=c.DEV_CUDA_VISIBLE_DEVICES,
                                                          server=c.DEV_SERVER,
                                                          params=model_parameter)
  return get_argrec_train_data, elmo_threshold_model_fn, model_parameter, eval_hook_listener


def get_multitask_run_config():
  logger.info("Launching multitask model")
  model_parameter = {"learning_rate": LEARNING_RATE,
                     "tune_embeddings": TUNE_EMBEDDINGS,
                     "shared_private": SHARED_PRIVATE,
                     "impose_orthogonality": IMPOSE_ORTHOGONALITY,
                     "lambda": LAMBDA,
                     "hidden_size": HIDDEN_SIZE,
                     "pretrained_emb_de": c.PATH_FASTTEXT_DE,
                     "pretrained_emb_en": c.PATH_FASTTEXT_EN,
                     "pretrained_emb_fr": c.PATH_FASTTEXT_FR}
  get_multitask_train_data  = lambda: {"nmt_ende": get_nmt_data(c.nmt_ende_tfrecord_train, is_train=True, epochs=EPOCHS,
                                                                cut_final_batch=False),
                                       "nmt_enfr": get_nmt_data(c.nmt_enfr_tfrecord_train, is_train=True, epochs=EPOCHS,
                                                                cut_final_batch=False),
                                       "nli": get_nli_data(c.nli_tfrecord_train, epochs=EPOCHS, is_train=True,
                                                           cut_final_batch=False)}
  eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=MODEL_HOME,
                                                          path_eval_script=c.PATH_EVAL_MULTITASK_SCRIPT,
                                                          server=DEV_SERVER,
                                                          gpu_fraction=DEV_GPU_FRACTION,
                                                          cuda_visible_devices=DEV_CUDA_VISIBLE_DEVICES,
                                                          params=model_parameter)
  return get_multitask_train_data, multitask_model_fn, model_parameter, eval_hook_listener


def get_treccar_run_config():
  logger.info("Launching Trec-CAR model")
  get_train_data = lambda: get_treccar_data(path="/home/gglavas/data/trec/treccar_train_big.tfrecord", contrastive=True)
  model_parameter = {"learning_rate": LEARNING_RATE,
                     "contrastive": "hinge",
                     "num_filters": 10,
                     "mlp_layers": [1],
                     "out_channels": 2,
                     "k_max_pooling": 2,
                     "fixed_qlen": 10,
                     "fixed_plen": 100,
                     "tune_embeddings": True}
  eval_hook_listener = EvalRoutineCheckpointSaverListener(model_dir=MODEL_HOME,
                                                          path_eval_script=c.PATH_EVAL_SCRIPT,
                                                          data_path="/home/gglavas/data/trec/treccar_dev.tfrecord",
                                                          input_fn=get_treccar_data,
                                                          model_fn=multichannel_model_fn,
                                                          gpu_fraction=DEV_GPU_FRACTION,
                                                          cuda_visible_devices=DEV_CUDA_VISIBLE_DEVICES,
                                                          server=DEV_SERVER,
                                                          params=model_parameter)
  return get_train_data, multichannel_model_fn, model_parameter, eval_hook_listener


def run_model(get_model_run_config):
  # LoggingTensorHook inside estimator writs tf.logging.info every 100s, change log-level to avoid this,
  tf.logging.set_verbosity(tf.logging.ERROR)
  hooks = []
  if c.DO_TENSORBOARD_DEBUG:
    hooks.append(tf_debug.TensorBoardDebugHook("dws-07:6064"))
    hooks.append(tf.train.ProfilerHook(save_steps=2, output_dir=MODEL_HOME))
    # hooks.append(MetadataHook(save_steps=1, output_dir= "/home/rlitschk/tmp"))

  get_train_data, model_fn, model_parameter, eval_hook_listener = get_model_run_config()
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=get_tf_run_config(), params=model_parameter)
  if c.DO_EVAL:
    logger.info("Async eval enabled.")
    res = estimator.train(input_fn=get_train_data, saving_listeners=[eval_hook_listener], hooks=[eval_hook_listener])
  else:
    logger.info("Run training only.")
    res = estimator.train(get_train_data)
  logger.info("Train (-eval) done, results (%s): %s" % (timer.pprint_lap(), str(res)))


def run_model_train_dev_test(get_model_run_config):
  # LoggingTensorHook inside estimator writs tf.logging.info every 100s, change log-level to avoid this,
  tf.logging.set_verbosity(tf.logging.ERROR)
  hooks = []
  if c.DO_TENSORBOARD_DEBUG:
    hooks.append(tf_debug.TensorBoardDebugHook("dws-07:6064"))
    hooks.append(tf.train.ProfilerHook(save_steps=2, output_dir=MODEL_HOME))
    # hooks.append(MetadataHook(save_steps=1, output_dir= "/home/rlitschk/tmp"))

  get_train_data, model_fn, model_parameter, eval_hook_listener, get_test_data = get_model_run_config()
  estimator = tf.estimator.Estimator(model_fn=model_fn, config=get_tf_run_config(), params=model_parameter)
  if c.DO_EVAL:
    logger.info("Async eval enabled.")
    res = estimator.train(input_fn=get_train_data, saving_listeners=[eval_hook_listener], hooks=[eval_hook_listener])
  else:
    logger.info("Run training only.")
    res = estimator.train(get_train_data)
  logger.info("Train (-eval) done, results (%s): %s" % (timer.pprint_lap(), str(res)))
  # TODO: Check with robert whether this is okay
  logger.info("Launching final test.")
  res = estimator.predict(input_fn=get_test_data)
  print(res)

def run_senteval():
  process_spec = [c.PYTHON_RUNTIME, c.PATH_SENTEVAL_SCRIPT, "--cuda_visible_devices", DEV_CUDA_VISIBLE_DEVICES,
                  "--model_home", MODEL_HOME + "eval/best_model/"]
  command = " ".join(process_spec)
  logger.info("Launch SentEval evaluation %s" % " ".join(process_spec))
  process = run_ssh_or_shell_command(command, DEV_SERVER)
  minute = 60
  process.communicate(timeout=180*minute)
  logger.info("SentEval evaluation done, duration: %s" % timer.pprint_lap())


def main():
  # log_specs(logger)
  run_model(get_argrec_run_config)
  #run_model(get_treccar_run_config)
  # run_multitask_model()
  #run_senteval()
  logger.info("Experiment done %s" % timer.pprint_stop())


if __name__ == "__main__":
  main()

# TODO: include dummy process in singletask training
# TODO: write module determining how much of gpu capacity is needed for a given model + max seq length
# TODO: write wrapper around eval processes such that they don#t need to manually occupy and return a dummy gpu pid

# TODO: combine tf logging and python logging into same output dir
# TODO: try out efficient GRU implementation!!!!
# TODO: include beamsearch decoder

# TODO: include option to train embedding matrix from scratch instead of GloVe/w2v embeddings
# TODO: add get parameters module for both main and eval routine + add cuda visible devices main param
# TODO: when constructing tfrecords: do lookup strategy with lowercase lookup
# TODO: rename maybe accuracy to NLI accuracy?

# self._hooks.pop(0)
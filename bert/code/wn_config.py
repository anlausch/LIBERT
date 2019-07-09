import os
# GPU allocations
#TRAIN_CUDA_VISIBLE_DEVICES = "0"  # "0"
#TRAIN_CUDA_GPU_FRAC = 1.0 # percentage of GPU memory that should be occupied
DEV_SERVER = "dws-04.informatik.uni-mannheim.de" # requires shell script in $PATH that connects to dws-0X, and host is added to known hosts
DEV_CUDA_VISIBLE_DEVICES = "1"
DEV_CUDA_GPU_FRAC = 1.0

DO_EVAL = True
DO_TENSORBOARD_DEBUG = False
TENACITY = 5  # number of steps to wait before early stopping, set None to disable

if os.name == "nt":
    # Python runtimes used for async evaluations
    PYTHON_HOME = "~/Anaconda3/"
    PYTHON_RUNTIME = PYTHON_HOME + "python"
else:
    # Python runtimes used for async evaluations
    PYTHON_HOME = "/home/anlausch/miniconda3/"
    PYTHON_RUNTIME = PYTHON_HOME + "bin/python"

if os.name == "nt":
    PROJECT_HOME = "C:\\Users\\anlausch\\PycharmProjects\\replant\\"
else:
    PROJECT_HOME = "/work/anlausch/replant/"

STANDARD_INPUT_TRAIN = ""
WN_INPUT_TRAIN = ""

STANDARD_INPUT_DEV=PROJECT_HOME + "bert/data/bert_text8.tfrecord"
WN_INPUT_DEV=PROJECT_HOME + "bert/data/wordnet_small.tfrecords"

BERT_CONFIG_FILE=PROJECT_HOME + "bert/code/bert_config_wn.json"

# Do not change this
PATH_EVAL_SCRIPT = PROJECT_HOME + "bert/code/eval_routine_st_wordnet_downstream.py"
PATH_EVAL_MULTITASK_SCRIPT = PROJECT_HOME + "bert/code/eval_routine_mt_wordnet.py"
PATH_DUMMY_PROCESS_SCRIPT = PROJECT_HOME + "util/gputil.py"


import os

EPOCHS = -1
HIDDEN_SIZE = 4096 # 2048 # 4000
EMBEDDING_SIZE = 300 # 512 # todo: remove this and infer it wherever its used
BATCH_SIZE = 8 # 24
VOCAB_SIZE = 30000 # 30000, 200000 doesn't work for nmt
MAX_SEQ_LEN = 100  # nmt models only
LEARNING_RATE = 1e-4 # 2e-3

# Orthogonality loss parameters, care only for multitask learning
LAMBDA = 0.5
IMPOSE_ORTHOGONALITY = False
TUNE_EMBEDDINGS = True
SHARED_PRIVATE = False

# GPU allocations
TRAIN_CUDA_VISIBLE_DEVICES = "0"  # "0"
TRAIN_CUDA_GPU_FRAC = 1.0 # percentage of GPU memory that should be occupied
DEV_SERVER = "dws-06.informatik.uni-mannheim.de" # requires shell script in $PATH that connects to dws-0X, and host is added to known hosts
DEV_CUDA_VISIBLE_DEVICES = "1"
DEV_CUDA_GPU_FRAC = 1.0

DO_EVAL = True
DO_TENSORBOARD_DEBUG = False
TENACITY = 5  # number of steps to wait before early stopping, set None to disable


# Tensorflow Estimator API parameters
SAVE_SUMMARY_STEPS = 10
LOG_STEP_COUNT_STEPS = 100000000
SAVE_CHECKPOINT_STEPS = 100
SAVE_CHECKPOINT_SECS = None

if os.name == "nt":
    PROJECT_HOME = "C:\\Users\\anlausch\\PycharmProjects\\replant\\"
else:
    PROJECT_HOME = "/work/anlausch/replant/"
LOG_FILE_NAME = "train_eval.log"
#MODEL_HOME = PROJECT_HOME + "check_dir/"
# SentEval parameters
PATH_SENTEVAL_SCRIPT = PROJECT_HOME + "sent_eval.py"
PATH_SENTEVAL_DATA = "/home/rlitschk/projects/SentEval/data"
# PATH_SENTEVAL_RESULTS = MODEL_HOME + "senteval_resutls.txt"

# if both SAVE_CHECKPOINT_STEPS and SAVE_CHECKPOINT_SECS are set to none SAVE_CHECKPOINT_SECS defaults to 600
assert bool(SAVE_CHECKPOINT_SECS) ^ bool(SAVE_CHECKPOINT_STEPS)  # xor, only one of the two can be set

if os.name == "nt":
    # Python runtimes used for async evaluations
    PYTHON_HOME = "~/Anaconda3/"
    PYTHON_RUNTIME = PYTHON_HOME + "python"
else:
    # Python runtimes used for async evaluations
    PYTHON_HOME = "/home/anlausch/miniconda3/"
    PYTHON_RUNTIME = PYTHON_HOME + "bin/python"

# TF Record serialization configuration parameters
PATH_TF_RECORDS = "/home/rlitschk/tfrecords_fromFastText/"
PATH_EMBEDDINGS = "/home/rlitschk/embeddings/"
# PATH_VOCABULARIES = "/home/rlitschk/data/vocabularies/no_stags/"
PATH_VOCABULARIES = "/home/rlitschk/vocabularies/fastText_vocabularies/"
DATA = "/home/rlitschk/data/"

"""
Paths for pretrained embeddings, needed for running models (used in model_fn)
"""
PATH_GLOVE300 = PATH_EMBEDDINGS + "glove.840B.300d.txt"
PATH_FASTTEXT_EN = PATH_EMBEDDINGS + "fasttext/wiki.en.vec"
PATH_FASTTEXT_DE = PATH_EMBEDDINGS + "fasttext/wiki.de.vec"
PATH_FASTTEXT_FR = PATH_EMBEDDINGS + "fasttext/wiki.fr.vec"

"""
Paths for vocabulary files, needed for creating tfrecords (words replaced by vocab id)
"""
PATH_EN_VOCAB_GLOVE300 = PATH_VOCABULARIES + "glove_vocab.pickle"
PATH_EN_VOCAB = PATH_VOCABULARIES + "en_vocab_%s.pickle" % str(VOCAB_SIZE)
PATH_DE_VOCAB = PATH_VOCABULARIES + "de_vocab_%s.pickle" % str(VOCAB_SIZE)
PATH_FR_VOCAB = PATH_VOCABULARIES + "fr_vocab_%s.pickle" % str(VOCAB_SIZE)

"""
Paths of NLI data files
"""
# Do not change this
PATH_EVAL_SCRIPT = PROJECT_HOME + "eval_routine_singletask.py"
PATH_EVAL_MULTITASK_SCRIPT = PROJECT_HOME + "eval_routine_multitask.py"
PATH_DUMMY_PROCESS_SCRIPT = PROJECT_HOME + "util/gputil.py"

snli_src_prefix = DATA + "snli_1.0/"
snli_train = snli_src_prefix + "snli_1.0_train.jsonl"
snli_dev = snli_src_prefix + "snli_1.0_dev.jsonl"
snli_test = snli_src_prefix + "snli_1.0_test.jsonl"

mnli_src_prefix = DATA + "multinli_1.0/"
mnli_train = mnli_src_prefix + "multinli_1.0_train.jsonl"
mnli_dev = mnli_src_prefix + "multinli_1.0_dev_matched.jsonl"

nli_tfrecord_train = PATH_TF_RECORDS + "nli_train.tfrecord"
nli_tfrecord_dev = PATH_TF_RECORDS + "nli_dev.tfrecord"
nli_tfrecord_test = PATH_TF_RECORDS + "nli_test.tfrecord"

"""
Paths of NMT data files
"""
# Train data
europarl_prefix = DATA + "Europarlv7/training/europarl-v7."
europarl_en = europarl_prefix + "de-en.en"
europarl_de = europarl_prefix + "de-en.de"
europarl_fr = europarl_prefix + "fr-en.fr"

# Dev data
newstest_prefix = DATA + "mt_dev/dev/"
newstest_en = newstest_prefix + "newstest2013.en"
newstest_de = newstest_prefix + "newstest2013.de"
newstest_fr = newstest_prefix + "newstest2013.fr"

nmt_ende_tfrecord_train = PATH_TF_RECORDS + "nmt_train_ende.tfrecord"
nmt_enfr_tfrecord_train = PATH_TF_RECORDS + "nmt_train_enfr.tfrecord"
nmt_ende_tfrecord_dev = PATH_TF_RECORDS + "nmt_dev_ende.tfrecord"
nmt_enfr_tfrecord_dev = PATH_TF_RECORDS + "nmt_dev_enfr.tfrecord"

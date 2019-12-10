#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
INPUT_FILE_STANDARD=".../wiki-en-pre2-correct.tfrecord"
INPUT_FILE_WORDNET=".../syn_hyp_constraints.tfrecord"
OUTPUT_DIR=".../poc_over_time/wn_binary_16_longer"
NUM_TRAIN_STEPS=4000000
BERT_BASE_DIR=""
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
WN_MODEL_VARIANT="WN_PAIRS_BINARY"


python run_pretraining_libert.py \
--input_file_standard=$INPUT_FILE_STANDARD \
--input_file_wn=$INPUT_FILE_WORDNET \
--output_dir=$OUTPUT_DIR \
--do_train=True \
--do_eval=True \
--multitask=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=16 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--save_checkpoints_steps=200000 \
--wn_model_variant=$WN_MODEL_VARIANT
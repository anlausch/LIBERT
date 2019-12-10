#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
INPUT_FILE_STANDARD=".../wiki-en-pre2-correct.tfrecord"
OUTPUT_DIR=".../base_16_longer"
NUM_TRAIN_STEPS=2000000
BERT_BASE_DIR=""
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json

python run_pretraining.py \
--input_file=$INPUT_FILE_STANDARD \
--output_dir=$OUTPUT_DIR \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=16 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--save_checkpoints_steps=200000
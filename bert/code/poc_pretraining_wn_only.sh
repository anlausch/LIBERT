#!/usr/bin/env bash
INPUT_FILE_STANDARD="/work/anlausch/wiki-en-pre2.tfrecord"
INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_identifiable.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_full.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_paired_identifiable.tfrecord"
#OUTPUT_DIR_STANDARD="/work/anlausch/replant/bert/pretraining/poc_wn_only/basic"
OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_wn_only/rescal_lm_50k"
#OUTPUT_DIR_WN2="/work/anlausch/replant/bert/pretraining/poc_wn_only/bert50k"
NUM_TRAIN_STEPS=50000

export CUDA_VISIBLE_DEVICES=2



#python run_pretraining_wordnet.py \
#--input_file_standard=$INPUT_FILE_STANDARD \
#--input_file_wn=$INPUT_FILE_WN \
#--output_dir=$OUTPUT_DIR_WN \
#--multitask=False \
#--do_train=True \
#--do_eval=True \
#--bert_config_file=./bert_config_wn.json \
#--train_batch_size=32 \
#--eval_batch_size=8 \
#--max_seq_length=128 \
#--max_predictions_per_seq=20 \
#--num_train_steps=$NUM_TRAIN_STEPS \
#--num_warmup_steps=1000 \
#--learning_rate=2e-5 \
#--max_eval_steps=1000 \
#--wn_model_variant=RESCAL

python run_pretraining_wordnet.py \
--input_file_standard=$INPUT_FILE_STANDARD \
--input_file_wn=$INPUT_FILE_WN \
--output_dir=$OUTPUT_DIR_WN \
--multitask=False \
--do_train=True \
--do_eval=True \
--bert_config_file=./bert_config_wn.json \
--train_batch_size=32 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--wn_model_variant=RESCAL
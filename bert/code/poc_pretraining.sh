#!/usr/bin/env bash
echo "script started"
#while gpusage | grep -q 'dws07    0'; do
#   echo "gpu still occupied"
#   sleep 120
#done
echo "starting process now";
export CUDA_VISIBLE_DEVICES=1
INPUT_FILE_STANDARD="/work/anlausch/wiki-en-pre2-correct.tfrecord"
INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_binary_new.tfrecord"
OUTPUT_DIR="/work/anlausch/replant/bert/pretraining/poc_over_time/wn_binary"
NUM_TRAIN_STEPS=400000
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
WN_MODEL_VARIANT="BERT"
BERT_CONFIG=./bert_config_wn.json

#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_full.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_identifiable.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_multiclass.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_paired_identifiable.tfrecord"
#OUTPUT_DIR_STANDARD="/work/anlausch/replant/bert/pretraining/poc_50k/basic_new"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/rescal_lm_double"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/wn_multiclass_double"
#BERT_CONFIG=./bert_config_wn_segments.json
#WN_MODEL_VARIANT="BERT_MULTICLASS"
#BERT_CONFIG=$BERT_BASE_DIR/bert_config.json


python run_pretraining_wordnet.py \
--input_file_standard=$INPUT_FILE_STANDARD \
--input_file_wn=$INPUT_FILE_WN \
--output_dir=$OUTPUT_DIR_WN \
--multitask=True \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=32 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--wn_model_variant=$WN_MODEL_VARIANT

#--NUM_MT_TRAIN_STEPS=$(($NUM_TRAIN_STEPS*2-18000))
#echo $NUM_MT_TRAIN_STEPS
#python run_pretraining_wordnet.py \
#--input_file_standard=$INPUT_FILE_STANDARD \
#--input_file_wn=$INPUT_FILE_WN \
#--output_dir=$OUTPUT_DIR_WN \
#--multitask=True \
#--do_train=True \
#--do_eval=True \
#--bert_config_file=$BERT_CONFIG \
#--train_batch_size=32 \
#--eval_batch_size=8 \
#--max_seq_length=128 \
#--max_predictions_per_seq=20 \
#--num_train_steps=$NUM_MT_TRAIN_STEPS \
#--num_warmup_steps=1000 \
#--learning_rate=2e-5 \
#--max_eval_steps=1000 \
#--wn_model_variant=$WN_MODEL_VARIANT \
#--init_checkpoint=$OUTPUT_DIR_WN/model.ckpt-18000
#!/usr/bin/env bash
INPUT_FILE_STANDARD="/work/anlausch/wiki-en-pre2.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_full.tfrecord"
INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_identifiable.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_multiclass.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_paired_identifiable.tfrecord"
#OUTPUT_DIR_STANDARD="/work/anlausch/replant/bert/pretraining/poc_50k/basic"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/rescal_lm_double"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/wn_multiclass_double"
NUM_TRAIN_STEPS=50000
WN_MODEL_VARIANT="RESCAL_LM"
BERT_CONFIG=./bert_config_wn_small.json

export CUDA_VISIBLE_DEVICES=0

#python run_pretraining_wordnet.py \
#--input_file_standard=$INPUT_FILE_STANDARD \
#--input_file_wn=$INPUT_FILE_WN \
#--output_dir=$OUTPUT_DIR_STANDARD \
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
#--max_eval_steps=1000
NUM_MT_TRAIN_STEPS=$(($NUM_TRAIN_STEPS*2))
echo $NUM_MT_TRAIN_STEPS

python run_pretraining_wordnet.py \
--input_file_standard=$INPUT_FILE_STANDARD \
--input_file_wn=$INPUT_FILE_WN \
--output_dir=$OUTPUT_DIR_WN \
--multitask=True \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=16 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_MT_TRAIN_STEPS \
--num_warmup_steps=1000 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--wn_model_variant=$WN_MODEL_VARIANT \
--init_checkpoint=$OUTPUT_DIR_WN/model.ckpt-6000
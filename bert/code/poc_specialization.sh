#!/usr/bin/env bash
#while gpusage | grep -q 'dws07    0'; do
#   echo "gpu still occupied.."
#   sleep 30
#done
#echo "starting process now";

#INPUT_FILE_STANDARD="/work/anlausch/wiki-en-pre2.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_full.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_identifiable.tfrecord"
#INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_paired_identifiable.tfrecord"
#OUTPUT_DIR_STANDARD="/work/anlausch/replant/bert/pretraining/poc_50k/basic"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
#OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/rescal_lm_double"
#CHECKPOINT_DIR="/work/anlausch/replant/bert/data/BERT_base"
INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_multiclass_normal_segs_new.tfrecord"
OUTPUT_DIR_WN="/work/anlausch/replant/bert/pretraining/poc_50k/specialization/wn_multiclass_new"
CHECKPOINT_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
CHECKPOINT=$CHECKPOINT_DIR/bert_model.ckpt
BERT_CONFIG=$CHECKPOINT_DIR/bert_config.json
NUM_TRAIN_STEPS=50000
WN_MODEL_VARIANT="BERT_MULTICLASS"

export CUDA_VISIBLE_DEVICES=1

python run_pretraining_wordnet.py \
--input_file_wn=$INPUT_FILE_WN \
--output_dir=$OUTPUT_DIR_WN \
--multitask=False \
--do_train=True \
--do_eval=True \
--bert_config_file=$BERT_CONFIG \
--train_batch_size=32 \
--eval_batch_size=8 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=$NUM_TRAIN_STEPS \
--num_warmup_steps=0 \
--learning_rate=2e-5 \
--max_eval_steps=1000 \
--wn_model_variant=$WN_MODEL_VARIANT \
--init_checkpoint=$CHECKPOINT
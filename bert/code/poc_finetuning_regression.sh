#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt

#export BERT_STANDARD_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/basic_new"
#export BERT_WORDNET_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn"
#export BERT_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_double_new"
#export RESCAL_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
BERT_POSTSPEC_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/specialization/wn_multiclass_new"

OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_50k"

#export VOCAB_EXTENDED_DIR="/work/anlausch/replant/bert/data/vocab_extended.txt"
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
#CHECKPOINT=$BERT_BASE_DIR/bert_model.ckpt
#CHECKPOINT=$BERT_STANDARD_DIR/model.ckpt-50000
CHECKPOINT=$BERT_POSTSPEC_DIR/model.ckpt-50000
#export BERT_CONFIG="/work/anlausch/replant/bert/code/bert_config_wn.json"
#CHECKPOIMNT=$RESCAL_WORDNET_DOUBLE_DIR/model.ckpt-100000

for task_name in "STSB" ; do #"RTE" "MRPC" "CoLA" "SST2" "QNLI" "QQP" "MNLI" ; do
    echo $task_name
    export GLUE_DATA="$GLUE_DIR/$task_name"

    python run_regression_wordnet.py   \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=true \
    --do_early_stopping=false \
    --data_dir=$GLUE_DATA \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$CHECKPOINT\
    --max_seq_length=128 \
    --train_batch_size="[16,32]" \
    --learning_rate="[2e-5, 3e-5]" \
    --num_train_epochs="[3,4]" \
    --output_dir=$OUTPUT_DIR/specialization/wn_multiclass_new/$task_name
done

#for task_name in "QQP" "QNLI" "RTE"; do #"SST2"; do #"MRPC" "MNLI" "CoLA"; do
#    echo $task_name
#    export GLUE_DATA="$GLUE_DIR/$task_name"
#
#    python run_classifier_wordnet.py   \
#    --task_name=$task_name \
#    --do_train=true \
#    --do_eval=true \
#    --data_dir=$GLUE_DATA \
#    --vocab_file=$VOCAB_DIR \
#    --bert_config_file=$BERT_CONFIG \
#    --init_checkpoint=$BERT_STANDARD_DIR/model.ckpt-50000 \
#    --max_seq_length=128 \
#    --train_batch_size="[16, 32]" \
#    --learning_rate="[5e-5, 3e-5, 2e-5]" \
#    --num_train_epochs="[3,4]" \
#    --output_dir=$OUTPUT_DIR/base/$task_name
#
#    python run_classifier_wordnet.py   \
#    --task_name=$task_name \
#    --do_train=true \
#    --do_eval=true \
#    --data_dir=$GLUE_DATA \
#   --vocab_file=$VOCAB_DIR \
#    --bert_config_file=$BERT_CONFIG \
#    --init_checkpoint=$BERT_WORDNET_DIR/model.ckpt-50000 \
#    --max_seq_length=128 \
#    --train_batch_size="[16, 32]" \
#    --learning_rate="[5e-5, 3e-5, 2e-5]" \
#    --num_train_epochs="[3,4]" \
#    --output_dir=$OUTPUT_DIR/wordnet/$task_name
#
#done
#
#
#for task_name in "QQP" "MNLI"; do
#    echo $task_name
#    export GLUE_DATA="$GLUE_DIR/$task_name"
#
#    python run_classifier_wordnet.py   \
#    --task_name=$task_name \
#    --do_train=true \
#    --do_eval=true \
#    --data_dir=$GLUE_DATA \
#    --vocab_file=$VOCAB_DIR \
#    --bert_config_file=$BERT_CONFIG \
#    --init_checkpoint=$BERT_WORDNET_DOUBLE_DIR/model.ckpt-100000 \
#    --max_seq_length=128 \
#    --train_batch_size="[16, 32]" \
#    --learning_rate="[5e-5, 3e-5, 2e-5]" \
#    --num_train_epochs="[3,4]" \
#    --output_dir=$OUTPUT_DIR/wordnet_double/$task_name
#
#done
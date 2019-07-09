#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
export BERT_STANDARD_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/basic"
#export BERT_WORDNET_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn"
export BERT_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_double"
export RESCAL_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
export OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_50k/early_stopping/"

export VOCAB_EXTENDED_DIR="/work/anlausch/replant/bert/data/vocab_extended.txt"
export VOCAB_DIR="/work/anlausch/replant/bert/data/vocab.txt"
export BERT_CONFIG="/work/anlausch/replant/bert/code/bert_config_wn.json"

for task_name in "MRPC" ; do
    echo $task_name
    export GLUE_DATA="$GLUE_DIR/$task_name"

    python run_classifier_wordnet.py   \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=false \
    --do_early_stopping=true \
    --data_dir=$GLUE_DATA \
    --vocab_file=$VOCAB_EXTENDED_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$BERT_WORDNET_DOUBLE_DIR/model.ckpt-100000 \
    --max_seq_length=128 \
    --train_batch_size="[16, 32]" \
    --learning_rate="[2e-5, 3e-5, 5e-5]" \
    --num_train_epochs="[5]" \
    --output_dir=$OUTPUT_DIR/wordnet_double/$task_name

    python run_classifier_wordnet.py   \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=false \
    --do_early_stopping=true\
    --data_dir=$GLUE_DATA \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$BERT_STANDARD_DIR/model.ckpt-50000 \
    --max_seq_length=128 \
    --train_batch_size="[16, 32]" \
    --learning_rate="[5e-5, 3e-5, 2e-5]" \
    --num_train_epochs="[5]" \
    --output_dir=$OUTPUT_DIR/base/$task_name
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
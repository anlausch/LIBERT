#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_STANDARD_DIR="/work/anlausch/replant/bert/pretraining/poc_over_time/wn_binary"
OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary/mismatched"
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json

#for STEP in "20000" "40000" "60000" "80000" "100000" "120000" "140000" "160000" "180000" "200000" "220000" "240000" "260000" "280000" "300000" "320000" "340000" "360000" "380000" "400000" ; do
#for STEP in "400000" "120000" "160000" "200000" "240000" "280000" "320000" "360000" ; do
for STEP in "2000000"; do
    #TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MNLI_16_3e-05_4/model.ckpt-98175"
    #TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MNLI_16_2e-05_3/model.ckpt-73631"
    #TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MNLI_16_2e-05_4/model.ckpt-98175"
    TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MNLI_16_3e-05_3/model.ckpt-73631"
    for task_name in "MNLI"; do
        echo $task_name
        echo $CHECKPOINT

        GLUE_DATA="$GLUE_DIR/$task_name"

        python run_classifier_wordnet.py   \
        --task_name=$task_name \
        --do_train=false \
        --do_eval=true \
        --do_early_stopping=false \
        --data_dir=$GLUE_DATA \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$TRAINED_CLASSIFIER\
        --max_seq_length=128 \
        --matched=False \
        --original_model=False \
        --output_dir=$OUTPUT_DIR/${STEP}/${task_name} |& tee $OUTPUT_DIR/${STEP}/${task_name}.out
    done
done
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
GLUE_DATA="$GLUE_DIR"
OUTPUT_DIR="/work/anlausch/replant/bert/predictions/base/mnli_neu"

ROOT="/work/anlausch/replant/bert/finetuning/poc_over_time/"
#for config in "base_16_longer/1000000/CoLA_16_2e-05_3/model.ckpt-1603","CoLA" "base_16_longer/1000000/MRPC_16_2e-05_4/model.ckpt-917","MRPC" "base_16_longer/1000000/SST2_16_2e-05_4/model.ckpt-16837","SST2" \
#"base_16_longer/1000000/QNLI_16_3e-05_3/model.ckpt-20331","QNLI" "base_16_longer/1000000/RTE_16_3e-05_3/model.ckpt-466","RTE" "base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175","MNLI" \
#"base_16/1000000/QQP_16_3e-05_4/model.ckpt-90962","QQP"; do

#for config in "base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175","MNLI" "base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175","diagnostic" "base_16_longer/1000000/QNLIV2_16_3e-05_3/model.ckpt-19639","QNLIV2" "base_16/1000000/QQP_16_3e-05_4/model.ckpt-90962","QQP"; do
#for config in "base_16/1000000/QQP_16_3e-05_4/model.ckpt-90962","QQP"; do
for config in "base_16/1000000/MNLI_16_2e-05_3/model.ckpt-73631","MNLI"; do
    IFS=","
    set -- $config
    echo $1 and $2
    TASK=$2
    TRAINED_CLASSIFIER=${ROOT}${1}

    python run_classifier_wordnet.py \
      --task_name=${TASK} \
      --do_predict=true \
      --do_train=false \
      --do_eval=false \
      --data_dir=$GLUE_DIR/${TASK} \
      --vocab_file=$BERT_BASE_DIR/vocab.txt \
      --bert_config_file=$BERT_BASE_DIR/bert_config.json \
      --init_checkpoint=$TRAINED_CLASSIFIER \
      --do_early_stopping=false \
      --max_seq_length=128 \
      --original_model=True \
      --matched=False \
      --output_dir=${OUTPUT_DIR}


    python parse_predictions.py \
    --task=${TASK} \
    --input_path="/work/anlausch/replant/bert/predictions/base/mnli_neu_32_5e-05_3.0/test_results.tsv" \
    --output_path_root="/work/anlausch/replant/bert/predictions/base/mnli_neu_32_5e-05_3.0/"
done

#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/STSB_16_2e-05_4/model.ckpt-1437"
#TASK="STSB"
#python run_regression_wordnet.py \
#  --task_name=${TASK} \
#  --do_predict=true \
#  --do_train=false \
#  --do_eval=false \
#  --data_dir=$GLUE_DIR/${TASK} \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$TRAINED_CLASSIFIER \
#  --do_early_stopping=false \
#  --max_seq_length=128 \
#  --original_model=True \
#  --output_dir=${OUTPUT_DIR}

#python parse_predictions.py \
#  --task=${TASK} \
#  --input_path="/work/anlausch/replant/bert/predictions/base_32_5e-05_3.0/test_results.tsv" \
#  --output_path_root="/work/anlausch/replant/bert/predictions/base_32_5e-05_3.0/"


#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/CoLA_16_2e-05_3/model.ckpt-1603"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/STSB_16_2e-05_4/model.ckpt-1437"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/MRPC_16_2e-05_4/model.ckpt-917"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/SST2_16_2e-05_4/model.ckpt-16837"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/QNLI_16_3e-05_3/model.ckpt-20331"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16_longer/1000000/RTE_16_3e-05_3/model.ckpt-466"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/QQP_16_3e-05_4/model.ckpt-90962"
#TASK="MNLI"


#python run_classifier_wordnet.py \
#  --task_name=${TASK} \
#  --do_predict=true \
#  --do_train=false \
#  --do_eval=false \
#  --data_dir=$GLUE_DIR/${TASK} \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$TRAINED_CLASSIFIER \
#  --do_early_stopping=false \
#  --max_seq_length=128 \
#  --original_model=True \
#  --matched=True \
#  --output_dir=${OUTPUT_DIR}
#
#
#python parse_predictions.py \
#  --task=${TASK}




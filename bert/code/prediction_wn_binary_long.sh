#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
GLUE_DATA="$GLUE_DIR"
OUTPUT_DIR="/work/anlausch/replant/bert/predictions/wn_binary_mnli"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/CoLA_16_2e-05_3/model.ckpt-1603"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/stsb_first/STSB_16_2e-05_3/model.ckpt-1077"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MRPC_16_3e-05_3/model.ckpt-687"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/SST2_16_3e-05_3/model.ckpt-12627"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/QNLI_16_2e-05_4/model.ckpt-27109"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/RTE_16_2e-05_4/model.ckpt-622"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/MNLI_16_3e-05_4/model.ckpt-98175"
#TASK="MNLI"
#TASK="diagnostic"
#QQP is not optimized yet!!!!

ROOT="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/"
#for config in "CoLA_16_2e-05_3/model.ckpt-1603","CoLA" "MRPC_16_3e-05_3/model.ckpt-687","MRPC" "SST2_16_3e-05_3/model.ckpt-12627","SST2" \
#"QNLI_16_2e-05_4/model.ckpt-27109","QNLI" "RTE_16_2e-05_4/model.ckpt-622","RTE" "MNLI_16_3e-05_4/model.ckpt-98175","MNLI" \
#"QQP_16_2e-05_3/model.ckpt-68221","QQP"; do
#for config in  "MNLI_16_3e-05_4/model.ckpt-98175","MNLI" "QQP_16_2e-05_4/model.ckpt-90962","QQP"; do
#for config in  "MNLI_16_3e-05_4/model.ckpt-98175","diagnostic" "QNLIV2_16_3e-05_3/model.ckpt-26185","QNLIV2" "QQP_16_2e-05_4/model.ckpt-90962","QQP"; do
#for config in  "QNLIV2_16_3e-05_3/model.ckpt-19639","QNLIV2" "QQP_16_2e-05_4/model.ckpt-90962","QQP"; do
for config in "MNLI_16_3e-05_3/model.ckpt-73631","MNLI"; do
#for config in  "QQP_16_2e-05_4/model.ckpt-90962","QQP"; do
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
      --original_model=False \
      --matched=False \
      --output_dir=${OUTPUT_DIR}


    python parse_predictions.py \
    --task=${TASK} \
    --input_path="/work/anlausch/replant/bert/predictions/wn_binary_mnli_32_5e-05_3.0/test_results.tsv" \
    --output_path_root="/work/anlausch/replant/bert/predictions/wn_binary_mnli_32_5e-05_3.0/"
done

#
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/wn_binary_16_longer/2000000/stsb_first/STSB_16_2e-05_3/model.ckpt-1077"
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
#  --original_model=False \
#  --output_dir=${OUTPUT_DIR}
#
#python parse_predictions.py \
#  --task=${TASK}
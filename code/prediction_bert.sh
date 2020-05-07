#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR=""
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
GLUE_DATA="$GLUE_DIR"
OUTPUT_DIR=".../predictions/base/"

ROOT=".../finetuning/poc_over_time/"

for config in "base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175","MNLI" "base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175","diagnostic" "base_16_longer/1000000/QNLIV2_16_3e-05_3/model.ckpt-19639","QNLIV2" "base_16/1000000/QQP_16_3e-05_4/model.ckpt-90962","QQP"; do
    IFS=","
    set -- $config
    echo $1 and $2
    TASK=$2
    TRAINED_CLASSIFIER=${ROOT}${1}

    python run_classifier_libert.py \
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
      --matched=True \
      --output_dir=${OUTPUT_DIR}


    python parse_predictions.py \
    --task=${TASK} \
    --input_path=".../test_results.tsv" \
    --output_path_root="..."
done

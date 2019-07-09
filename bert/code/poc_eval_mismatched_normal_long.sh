#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/mismatched"
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json


STEP="1000000"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_3e-05_3/model.ckpt-73631"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_2e-05_3/model.ckpt-73631"
#TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_2e-05_4/model.ckpt-98175"
TRAINED_CLASSIFIER="/work/anlausch/replant/bert/finetuning/poc_over_time/base_16/1000000/MNLI_16_3e-05_4/model.ckpt-98175"
TASK="MNLI"
GLUE_DATA="$GLUE_DIR/${TASK}"

python run_classifier_wordnet.py   \
--task_name=$TASK \
--do_train=False \
--do_eval=true \
--do_early_stopping=false \
--data_dir=$GLUE_DATA \
--vocab_file=$VOCAB_DIR \
--bert_config_file=$BERT_CONFIG \
--init_checkpoint=$TRAINED_CLASSIFIER \
--max_seq_length=128 \
--original_model=True \
--matched=False \
--output_dir=$OUTPUT_DIR/${STEP}/${TASK} |& tee $OUTPUT_DIR/${STEP}/${TASK}.out

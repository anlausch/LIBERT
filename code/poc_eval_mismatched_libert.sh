#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=;
BERT_BASE_DIR=""
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_STANDARD_DIR=""
OUTPUT_DIR=""
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json


for STEP in "2000000"; do
TRAINED_CLASSIFIER="/libert/2000000/MNLI_16_3e-05_3/model.ckpt-73631"
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
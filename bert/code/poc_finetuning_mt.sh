#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_50k"
INPUT_FILE_WN="/work/anlausch/replant/bert/data/wordnet_multiclass_normal_segs_new.tfrecord"
BERT_BASE_DIR="/work/anlausch/replant/bert/data/BERT_base_new"
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json
CHECKPOINT=$BERT_BASE_DIR/bert_model.ckpt
WN_MODEL_VARIANT="BERT_MULTICLASS"


for task_name in "CoLA" ; do
    echo $task_name
    export GLUE_DATA="$GLUE_DIR/$task_name"

    python run_classifier_wordnet_mt.py   \
    --task_name=$task_name \
    --do_train=true \
    --do_eval=true \
    --do_early_stopping=false \
    --multitask=true \
    --data_dir=$GLUE_DATA \
    --input_file_wn=$INPUT_FILE_WN \
    --vocab_file=$VOCAB_DIR \
    --bert_config_file=$BERT_CONFIG \
    --init_checkpoint=$CHECKPOINT \
    --max_seq_length=128 \
    --train_batch_size="[16, 32]" \
    --learning_rate="[2e-5, 3e-5]" \
    --num_train_epochs="[3,4]" \
    --output_dir=$OUTPUT_DIR/mt_specialization_new2/wn_multiclass_rerun/$task_name \
    --wn_model_variant=$WN_MODEL_VARIANT
done
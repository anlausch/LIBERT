#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
BERT_BASE_DIR=""
VOCAB_DIR=$BERT_BASE_DIR/vocab.txt
BERT_STANDARD_DIR=".../wn_binary_16_longer"
OUTPUT_DIR=".../finetuning/poc_over_time/wn_binary_16_longer"
BERT_CONFIG=$BERT_BASE_DIR/bert_config.json

for STEP in "20000" "40000" "60000" "80000" "100000" "120000" "140000" "160000" "180000" "200000" "220000" "240000" "260000" "280000" "300000" "320000" "340000" "360000" "380000" "400000" ; do
    CHECKPOINT=${BERT_STANDARD_DIR}/model.ckpt-${STEP}


    # check whether there is a problem with the regression tasks
    for task_name in "STSB"; do
        echo $task_name
        export GLUE_DATA="$GLUE_DIR/$task_name"

        python run_regression_libert.py   \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_early_stopping=false \
        --data_dir=$GLUE_DATA \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$CHECKPOINT\
        --max_seq_length=128 \
        --train_batch_size="[16]" \
        --learning_rate="[2e-5, 3e-5]" \
        --num_train_epochs="[3,4]" \
        --original_model=False \
        --output_dir=$OUTPUT_DIR/${STEP}/${task_name}  |& tee $OUTPUT_DIR/${STEP}/${task_name}.out
    done

    for task_name in "CoLA" "MRPC" "RTE" "SST2" "QNLIV2" "MNLI" "QQP"; do
        echo $task_name
        echo $CHECKPOINT

        GLUE_DATA="$GLUE_DIR/$task_name"

        python run_classifier_libert.py   \
        --task_name=$task_name \
        --do_train=true \
        --do_eval=true \
        --do_early_stopping=false \
        --data_dir=$GLUE_DATA \
        --vocab_file=$VOCAB_DIR \
        --bert_config_file=$BERT_CONFIG \
        --init_checkpoint=$CHECKPOINT\
        --max_seq_length=128 \
        --train_batch_size="[16]" \
        --learning_rate="[2e-5, 3e-5]" \
        --num_train_epochs="[3,4]" \
        --original_model=False \
        --output_dir=$OUTPUT_DIR/${STEP}/${task_name} |& tee $OUTPUT_DIR/${STEP}/${task_name}.out
    done

done
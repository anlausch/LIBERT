#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=3
export BERT_STANDARD_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/basic"
export BERT_WORDNET_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn"
export BERT_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_double"
export RESCAL_WORDNET_DOUBLE_DIR="/work/anlausch/replant/bert/pretraining/poc_50k/wn_rescal_double"
export OUTPUT_DIR="/work/anlausch/replant/bert/finetuning/poc_50k"
export VOCAB_EXTENDED_DIR="/work/anlausch/replant/bert/data/vocab_extended.txt"
export VOCAB_DIR="/work/anlausch/replant/bert/data/vocab.txt"
export BERT_CONFIG="./bert_config_wn.json"
export SQUAD_DIR="/work/anlausch/squad2.0"

python run_squad.py \
  --vocab_file=$VOCAB_EXTENDED_DIR \
  --bert_config_file=$BERT_CONFIG \
  --init_checkpoint=$BERT_STANDARD_DIR/model.ckpt-50000 \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size="[8]" \
  --learning_rate="[3e-5]" \
  --num_train_epochs="[2.0]" \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR/squad2/base2/ \
  --version_2_with_negative=True

#python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUTPUT_DIR/squad/base2/_8_3e-05_3.0/predictions.json >> $OUTPUT_DIR/squad/base2/_8_3e-05_3.0/out.txt

python run_squad.py \
  --vocab_file=$VOCAB_EXTENDED_DIR \
  --bert_config_file=$BERT_CONFIG \
  --init_checkpoint=$BERT_WORDNET_DOUBLE_DIR/model.ckpt-100000 \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size="[8]" \
  --learning_rate="[3e-5]" \
  --num_train_epochs="[2.0]" \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=$OUTPUT_DIR/squad2/wordnet_double/ \
  --version_2_with_negative=True

#python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json $OUTPUT_DIR/squad/wordnet_double/_8_3e-05_3.0/predictions.json >> $OUTPUT_DIR/squad/wordnet_double/_8_3e-05_3.0/out.txt
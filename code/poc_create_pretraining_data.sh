#!/usr/bin/env bash

python create_pretraining_data.py \
--input_file="path to wikipedia" \
--output_file="path to output tfrecord" \
--vocab_file=/bert_base_uncased/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5

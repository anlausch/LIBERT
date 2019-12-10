#!/usr/bin/env bash

python create_pretraining_data.py \
--input_file=/work/anlausch/wiki-en-pre2.txt \
--output_file=/work/anlausch/wiki-en-pre2-correct.tfrecord \
--vocab_file=/work/anlausch/replant/bert/data/BERT_base_new/vocab.txt \
--do_lower_case=True \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--masked_lm_prob=0.15 \
--random_seed=12345 \
--dupe_factor=5

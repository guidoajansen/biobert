#!/usr/bin/env bash
train=true
eval=true
dotest=false
dataset="$1"
timestamp=$( date +%T )

export TPU_NAME=biobert-tpu
export BIOBERT_DIR=gs://biobert-bucket/biobert-pubmed-pmc
export NER_DIR=gs://biobert-bucket/data/$dataset
export OUTPUT_DIR=gs://biobert-bucket/output/

python biobert/run_ner.py \
	   --do_train=$train \
	   --do_eval=$eval \
	   --do_test=$dotest \
	   --use_tpu=true \
	   --vocab_file=$BIOBERT_DIR/vocab.txt \
	   --bert_config_file=$BIOBERT_DIR/bert_config.json \
	   --init_checkpoint=$BIOBERT_DIR/bert_model.ckpt \
	   --num_train_epochs=3.0 \
	   --data_dir=$NER_DIR/ \
	   --output_dir=$OUTPUT_DIR/$dataset_$timestamp \
	   --dataset=$dataset
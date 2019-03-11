#!/usr/bin/env bash
train=true
eval=true
dotest=true
timestamp=$( date +%s )
dataset="$1"
pretrained="$2"
epochs="$3"

export TPU_NAME=biobert-tpu
export PRETRAINED_DIR=gs://biobert-bucket/pre-trained/$pretrained
export NER_DIR=gs://biobert-bucket/data/$dataset
export OUTPUT_DIR=gs://biobert-bucket/output/$dataset/$timestamp
export OUTPUT_DIR=gs://biobert-bucket/export/$dataset/$timestamp

rm ./output/$dataset/*.txt

python biobert/run_ner.py \
	   --do_train=$train \
	   --do_eval=$eval \
	   --do_test=$dotest \
	   --use_tpu=true \
	   --vocab_file=$PRETRAINED_DIR/vocab.txt \
	   --bert_config_file=$PRETRAINED_DIR/bert_config.json \
	   --init_checkpoint=$PRETRAINED_DIR/bert_model.ckpt \
	   --num_train_epochs=$epochs \
	   --data_dir=$NER_DIR/ \
	   --output_dir=$OUTPUT_DIR/ \
	   --export_dir=$EXPORT_DIR/ \
	   --dataset=$dataset \
	   --pretrained=$pretrained

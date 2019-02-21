train:
	TPU_NAME='biobert-tpu'
	BIOBERT_DIR='gs://biobert-bucket/biobert-pubmed-pmc'
	NER_DIR='gs://biobert-bucket/data/i2b2/NCBI-disease'
	OUTPUT_DIR='gs://biobert-bucket/output/'

	python3 run_ner.py \
		--do_train=true \
		--do_eval=true \
		--do_predict=false \
		--user_tpu=true \
		--tpu_name=TPU_NAME \
		--vocab_file=$(BIOBERT_DIR)/vocab.txt \
		--bert_config_file=$(BIOBERT_DIR)/bert_config.json \
		--init_checkpoint=$(BIOBERT_DIR)/bert_model.ckpt \
		--num_train_epochs=10.0 \
		--data_dir=$(NER_DIR)/ \
		--output_dir=/tmp/bioner/

predict:
	export TPU_NAME=biobert-tpu
	export BIOBERT_DIR=gs://biobert-bucket/biobert-pubmed-pmc
	export NER_DIR=gs://biobert-bucket/data/i2b2/$dataset
	export OUTPUT_DIR=gs://biobert-bucket/output/

	python run_ner.py \
	--do_train=false \
	--do_eval=true \
	--do_predict=true \
	--vocab_file=${BIOBERT_DIR}/vocab.txt \
	--bert_config_file=${BIOBERT_DIR}/bert_config.json \
	--init_checkpoint=${BIOBERT_DIR}/bert_model.ckpt \
	--num_train_epochs=10.0 \
	--data_dir=${NER_DIR}/ \
	--output_dir=/tmp/bioner/



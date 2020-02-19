#!/bin/sh

export CUDA_VISIBLE_DEVICES=$4
export BERT_BASE_DIR=/mnt/nfs/baliabideLexikalak/wordEmbedings/eu/$1
export GLUE_DIR=/home/inaki/datuak/$3
export EPOCHS=4
export RESULT_OUTPUT_DIR_BASE="bert-base-output-e"$EPOCHS
for i in 1 2 3 4 5
do
    echo "*********************   TRAIN FOR RUN $i ***********************************************"
    bert-base-ner-train \
	-ner ner \
	-do_train True \
	-do_eval True \
	-do_predict True \
	-data_dir $GLUE_DIR \
	-vocab_file $BERT_BASE_DIR/vocab.txt \
	-bert_config_file $BERT_BASE_DIR/config.json \
	-init_checkpoint $BERT_BASE_DIR/$2 \
	-max_seq_length 128 \
	-batch_size 32 \
	-learning_rate 2e-5 \
	-num_train_epochs $EPOCHS.0 \
	-output_dir $GLUE_DIR/$RESULT_OUTPUT_DIR_BASE-$i \
        -device_map $4 \
        -verbose \
        -label_list $GLUE_DIR/manual_labels.txt
        #-do_lower_case \
	#-dropout_rate 0.9 	
	


    #-crf True \


    echo "*********************   END RUN $i ***********************************************"   
done

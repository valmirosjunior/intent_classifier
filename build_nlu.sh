#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "It's necessary to provide only the WORK DIR PATH, where are the split data for train and test"
fi


WORK_DIR=$1
#EMBEDDINGS=( bert_pt flair_pt glove lasbe use )
#
#for model in ${EMBEDDINGS[@]}
#do
#  WORK_DIR_MODEL="$WORK_DIR/$model"
#
#  echo $WORK_DIR_MODEL
#done


TRAINING_FILE="$WORK_DIR/training_data.yml"
TESTING_FILE="$WORK_DIR/test_data.yml"


echo 'activating nlu_env...'
cd src/nlu_builder
. venv_nlu/bin/activate


echo 'entering in rasa directory...'
cd rasa

echo 'training model....'
rasa train nlu -u $TRAINING_FILE

echo 'testing model....'
rasa test nlu --nlu $TESTING_FILE

mkdir -p "$WORK_DIR"

mv results $WORK_DIR
mv models $WORK_DIR
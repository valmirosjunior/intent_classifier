#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "It's necessary to provide only the WORK DIR PATH, where are the split data for train and test"
fi

WORK_DIR=$1
SUBFOLDER_LIST=( k100 k100_without_outliers k100_without_sentences_higher_than_median )
EMBEDDINGS=( bert_pt flair_pt glove lasbe use )

echo 'activating nlu_env...'
cd src/nlu_builder
. venv_nlu/bin/activate

echo 'entering in rasa directory...'
cd rasa


for SUBFOLDER in ${SUBFOLDER_LIST[@]}
do
  for EMBEDDING in ${EMBEDDINGS[@]}
  do
    WORK_DIR_NLU="$WORK_DIR/$SUBFOLDER/$EMBEDDING"

    echo $WORK_DIR_NLU

    TRAINING_FILE="$WORK_DIR_NLU/training_data.yml"
    TESTING_FILE="$WORK_DIR_NLU/test_data.yml"

    echo 'training model....'
    rasa train nlu -u $TRAINING_FILE

    echo 'testing model....'
    rasa test nlu --nlu $TESTING_FILE

    mkdir -p "$WORK_DIR_NLU"

    mv results $WORK_DIR_NLU
    mv models $WORK_DIR_NLU
  done
done

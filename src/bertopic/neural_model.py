import json
import numpy as np
import os
import random as rn
import pandas as pd
import tensorflow as tf

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
# from pathlib import Path

from src.core import file_manager as fm


def build_model(embedding_dim, num_labels):
    model = Sequential()
    model.add(Dense(64, activation='softmax', input_shape=(embedding_dim,)))
    model.add(Dropout(0.1))
    model.add(Dense(num_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def get_metrics_by_average(y_true, y_pred, labels, average):
  return {
    'precision': precision_score(y_true, y_pred, labels=labels, average=average),
    'recall': recall_score(y_true, y_pred, labels=labels, average=average),
    'f1': f1_score(y_true, y_pred, labels=labels, average=average)
    }

def get_metrics_results(y_true, y_pred, labels):
  corect_predicts = np.equal(y_true, y_pred).sum()
  accuracy = corect_predicts / len(y_true)

  return {
    'accuracy': accuracy_score(y_true, y_pred),
    'simples_accuracy': accuracy,
    'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
    'weighted': get_metrics_by_average(y_true, y_pred, labels, 'weighted'),
    'macro': get_metrics_by_average(y_true, y_pred, labels, 'macro'),
    'micro': get_metrics_by_average(y_true, y_pred, labels, 'micro'),
  }






# create_train_test_data('bert_pt', df_to_use)

class NeuralModel:
    DEFAULT_RANDOM_SEED = 42

    INTENT_INDEXES_DICT = {
        'greeting': 0,
        'inform_medicine': 1,
        'inform_symptoms': 2,
        'request_inform': 3,
    }

    def __init__(self, embedding_name, variation = 'without_sentences_higher_than_median/', actor='patient'):
        self.embedding_name = embedding_name
        self.actor = actor
        self.random_seed = self.DEFAULT_RANDOM_SEED
        self.embedding_dir = fm.filename_from_data_dir(f'output/patient/bertopic/without_others_intent/{self.embedding_name}')
        self.variation = variation
        self.variation_dir = f'{self.embedding_dir}/{variation}'


    def describe_sentences(self):
        df = pd.read_csv(f'{self.variation_dir}annotated_sentences.csv')

        return df
    
    
    def run_pipeline(self):
        self.apply_seed()
        
        # variations = ['', 'without_outliers/', 'without_sentences_higher_than_median/']
        print(f"Split data for variation: {self.variation}")
        X_train, X_test, Y_train, Y_test = self.train_test_data()

        print("Build Model...")
        model = build_model(X_train.shape[1], Y_train.shape[1])

        print('training model')
        history = model.fit(X_train, Y_train, epochs=100, batch_size=64,validation_split=0.1, verbose=0,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

        # accr = model.evaluate(X_test,Y_test)
        # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
        
        print("Save Model...")
        model.save(f'{self.variation_dir}model.h5')

        predictions = model.predict(X_test)
        labels = np.array(list(range(Y_train.shape[1])))

        y_true = np.array([np.argmax(prediction) for prediction in Y_test])
        y_pred = np.array([np.argmax(prediction) for prediction in predictions])
        
        print('get metrics')
        metrics = get_metrics_results(y_true, y_pred, labels)           
        print(json.dumps(metrics, indent=4))


    def apply_seed(self, verbosity=True):
        if verbosity:
            print('Applying seed')
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        np.random.seed(self.random_seed)
        rn.seed(self.random_seed)
        tf.random.set_seed(self.random_seed )

    
    def train_test_data(self):
        df_train, df_test = self.create_train_test_data()

        # X_train = np.array(df_train['embeddings'].map(lambda x: np.array(json.loads(x)[0])).to_list())
        X_train = np.array(df_train['embeddings'].map(lambda x: np.array(x[0])).to_list())
        Y_train = pd.get_dummies(df_train['intent']).values

        # X_test = np.array(df_test['embeddings'].map(lambda x: np.array(x[0])).to_list())
        # X_test = np.array(df_test['embeddings'].map(lambda x: np.array(json.loads(x)[0]))
        X_test = np.array(df_test['embeddings'].map(lambda x: np.array(x[0])).to_list())        
        Y_test = pd.get_dummies(df_test['intent']).values

        return X_train, X_test, Y_train, Y_test


    def create_train_test_data(self):
        file_name_of_variation = f'{self.variation_dir}annotated_sentences.csv'
        df_annotated_for_variation = pd.read_csv(file_name_of_variation)
        df_to_merge_embeddings = df_annotated_for_variation[~df_annotated_for_variation['intent'].isin(['outliers', 'others'])]

        df = self.read_annotated_df_with_embeddings(df_to_merge_embeddings)
        print(f"The total of sentences is: {df.txt.count()}")

        df_without_validation = df[~df['txt'].isin(self.get_validation_data()['txt'])]
        print(f"The total of sentences after remove validation is: {df_without_validation.txt.count()}")

        
        df_train, df_test = train_test_split(df_without_validation, test_size=0.3, random_state=42)

        df_train.to_csv(f'{self.variation_dir}training_data.csv', index=False)
        df_test.to_csv(f'{self.variation_dir}test_data.csv', index=False)

        # setup_data.generate_nlu_file_from_df(df_train, f'{work_dir}/training_data.yml')
        # setup_data.generate_nlu_file_from_df(df_test, f'{work_dir}/test_data.yml')
        
        return df_train, df_test


    def read_annotated_df_with_embeddings(self, df_to_merge_embeddings):
        """
        Read the dataset with embeddings.

        :param str variation: The subfolder of data pass it like this: 'without_outilers/'
        pay attention at the end of variation, you must provide the '/'
        """

        df_embeddings = fm.read_json_of_dir(
            fm.filename_from_data_dir(
                f'embeddings/{self.embedding_name}/text_emb_{self.actor}.json'),
            lines=True
        )
        # df_with_embeddings = df_embeddings.drop('annotated_txt', axis=1)
        df_with_embeddings = df_embeddings


        df_merged = pd.merge(df_to_merge_embeddings, df_with_embeddings, on='txt', how='left')
        
        print("nan values:",df_merged[df_merged['embeddings'].isna()]['txt'])

        return df_merged.dropna()
    
    
    def get_validation_data(self):
        data_to_valid = pd.read_csv(fm.filename_from_data_dir(f'output/patient/bertopic/data_to_valid.csv'))

        return data_to_valid.drop('doubt', axis=1)


    def run_validation_pipeline(self):
        print('Loading validation data....')
        data_to_valid = self.get_validation_data()
        df_with_embeddings =  self.read_annotated_df_with_embeddings(data_to_valid)

        x_validation = np.array(df_with_embeddings['embeddings'].map(lambda x: np.array(x[0])).to_list())

        print(f'The embedding: {self.embedding_name} has a dimensionality of: {x_validation.shape[1]}')

        print('Loading model....')
        model = load_model(f'{self.variation_dir}model.h5')

        print('Running pridictions....')
        predictions = model.predict(x_validation)

        y_true = data_to_valid['intent_index'].to_numpy()
        y_pred = np.array([np.argmax(prediction) for prediction in predictions])
        labels = np.array(list(self.INTENT_INDEXES_DICT.values()))

        return get_metrics_results(y_true, y_pred, labels)



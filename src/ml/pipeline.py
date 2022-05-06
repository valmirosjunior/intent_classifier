from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.core import file_manager
from src.ml.h2o_helper import H2OHelper

dict_labels = {
    'inform': 1,
    'inform_symptoms': 2,
    'inform_medicine': 3,
    'greeting': 4,
    'request_inform': 5
}


def get_inverted_dict_labels():
    return {v: k for k, v in dict_labels.items()}


def get_annotated_df(embedding_name):
    annotated_filename = file_manager.filename_from_data_dir(
        f'output/patient/k100/{embedding_name}/annotated_sentences.csv')

    return pd.read_csv(annotated_filename)


def get_embedding_dfs(embedding_name):
    embeddings_paths = glob(
        f'{file_manager.filename_from_data_dir(f"models/{embedding_name}/text_emb_patient.json")}/*.json'
    )

    return file_manager.read_multiple_files(embeddings_paths)


def generate_dict_embedding_text(df_embeddings):
    return pd.Series(
        df_embeddings.embeddings.map(lambda x: np.array(x[0])).to_list(), index=df_embeddings.txt
    ).to_dict()


def transform_embeddings_in_dataframe(embeddings):
    vectors = [{f'V_{index:03d}': value for index, value in enumerate(embedding)} for embedding in embeddings]

    df_data = pd.DataFrame(data=vectors)

    return df_data


def generate_df_from_x_y(x_data, y_data):
    df_data = transform_embeddings_in_dataframe(x_data)

    df_data['label'] = y_data

    return df_data


class PipelineH2OManualLabelingData:
    def __init__(self, embedding_name='use'):
        self.embedding_name = embedding_name

        self.df_embeddings = get_embedding_dfs(self.embedding_name)
        self.dict_embeddings = generate_dict_embedding_text(self.df_embeddings)

        self.df_annotated = get_annotated_df('use')

        self.df_annotated['embeddings'] = self.df_annotated['txt'].map(self.dict_embeddings)

        self.df_data_to_train_model = pd.read_csv(
            file_manager.filename_from_data_dir('output/sentences_classifieds_of_all_distances.csv')
        )

        self.df_data_to_train_model['embeddings'] = self.df_data_to_train_model['txt'].map(self.dict_embeddings)

        self.df_data_to_train_model['label_index'] = self.df_data_to_train_model['label'].map(dict_labels)

        self.df_train, self.df_test = self.split_data()

        self.h2o_helper = H2OHelper(self.df_train, self.df_test)

    def split_data(self):
        x = self.df_data_to_train_model['embeddings'].to_numpy()
        y = self.df_data_to_train_model['label_index'].to_numpy()

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        df_train = generate_df_from_x_y(x_train, y_train)
        df_test = generate_df_from_x_y(x_test, y_test)

        return df_train, df_test

    def predict_labels(self):
        df_to_predict = transform_embeddings_in_dataframe(self.df_annotated['embeddings'].to_numpy())

        predicted_labels = self.h2o_helper.predict_labels(df_to_predict)

        self.df_annotated['old_intent'] = self.df_annotated['intent']
        self.df_annotated['label_index_predict'] = predicted_labels
        self.df_annotated['intent'] = self.df_annotated['label_index_predict'].map(get_inverted_dict_labels())

    def save_annotated_data(self):
        desired_columns = ['txt', 'label', 'distance', 'intent', 'annotated_txt']

        result_df = self.df_annotated[desired_columns]

        output_dir = Path(file_manager.filename_from_data_dir(
                f'output/h2o/input_data_classified_manual/{self.embedding_name}'
        ))

        output_dir.mkdir(parents=True, exist_ok=True)

        result_df.to_csv(f'{output_dir}/annotated_sentences.csv')

    def run(self):
        print('training model...')
        self.h2o_helper.train()

        print('testing model...')
        self.h2o_helper.test()

        print('saving model...')
        self.h2o_helper.save_model(suffix_output_dir=f'classified_manual/{self.embedding_name}')

        print('predicting labels...')
        self.predict_labels()

        print('saving data...')
        self.save_annotated_data()



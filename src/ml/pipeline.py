import pandas as pd

import numpy as np
from glob import glob
from src.core import file_manager

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


def generate_df_from_x_y(x_data, y_data):
    vectors = [{f'V_{index:03d}': value for index, value in enumerate(embedding)} for embedding in x_data]

    df_data = pd.DataFrame(data=vectors)

    df_data['label'] = y_data

    return df_data

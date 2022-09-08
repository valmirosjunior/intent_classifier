from glob import glob
from pathlib import Path

import pandas as pd

__PROJECT_DIR = str(Path(__file__).parent.parent.parent)


def get_project_dir():
    return __PROJECT_DIR


def filename_from_project_dir(filename):
    return f'{get_project_dir()}/{filename}'


def filename_from_data_dir(filename):
    return filename_from_project_dir(f'data/{filename}')

def create_dir_if_not_exists_on_data_dir(path_dir):
    output_dir = Path(filename_from_data_dir(path_dir))
    
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def read_json_of_dir(dir_path, actor='patient', **kwargs):
    filenames = glob(f'{dir_path}/*.json')

    filenames_ordered = sorted(filenames)

    data_frames = [pd.read_json(file, **kwargs) for file in filenames_ordered]

    df_embeddings = pd.concat(data_frames)

    df_annotated = pd.read_csv(filename_from_data_dir(f'output/{actor}/annotated_sentences.csv'))
    
    df_annotated['embeddings'] = df_embeddings['embeddings']

    return df_annotated

def read_annotated_df_with_embeddings(embedding_name, actor='patient', variation='without_others_intent/k100_without_sentences_higher_than_median'):
    df_embeddings = read_json_of_dir(
        filename_from_data_dir(
            f'embeddings/{embedding_name}/text_emb_{actor}.json'),
        lines=True
    )

    file_name_of_variation = filename_from_data_dir(
        f'output/{actor}/{variation}/{embedding_name}/annotated_sentences.csv'
    )

    df_annotated_for_variation = pd.read_csv(file_name_of_variation)


    df_merged = pd.merge(df_annotated_for_variation, df_embeddings.drop('annotated_txt', axis=1), on='txt', how='left')

    return df_merged

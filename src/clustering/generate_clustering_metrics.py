import json
import pandas as pd
from src.core import file_manager as fm
from src.clustering.pipeline_helper import PipelineHelper
from src.clustering.metrics_helper import get_clustering_metrics

MAP_EMBEDDING_K_VALUE = {
    'bert_pt': 91,
    'flair_pt': 95,
    'glove': 99,
    'lasbe': 81,
    'use': 82,
}


def get_metrics(data_helper):
    metrics_all = get_clustering_metrics(data_helper.df.label.to_numpy(), data_helper.get_embeddings())

    data_helper.remove_outlier_sentences()
    metrics_not_outliers = get_clustering_metrics(data_helper.df.label.to_numpy(), data_helper.get_embeddings())

    data_helper.remove_higher_than_median_sentences()
    metrics_not_median = get_clustering_metrics(data_helper.df.label.to_numpy(), data_helper.get_embeddings())

    data = {
        'variation': ['all', 'not_outliers', 'not_median'],
        'silhouette': [metrics_all[0], metrics_not_outliers[0], metrics_not_median[0]],
        'davies_bouldin': [metrics_all[1], metrics_not_outliers[1], metrics_not_median[1]]
    }

    return pd.DataFrame(data)


def annotate_intents(data_helper, intents_dictionary_filename):
    intents_dictionary_file = open(intents_dictionary_filename)

    dict_intents = {
        int(key): value
        for key, value in json.load(intents_dictionary_file).items()
    }

    data_helper.df['intent'] = data_helper.df['label'].map(dict_intents)

    return data_helper


def run_pipeline(embedding_name):
    subfolder = 'k100'

    pipeline = PipelineHelper(
        embedding_name=embedding_name,
        actor='patient',
        k=MAP_EMBEDDING_K_VALUE[embedding_name],
        sub_folder_k=subfolder
    )

    print('Run clustering')
    pipeline.run_clustering()

    data_helper = pipeline.data_helper

    print('Getting all_intents_metrics')
    all_intents_metrics_df = get_metrics(data_helper)

    data_helper.reset_df()

    work_dir = fm.filename_from_data_dir(f'output/patient/{subfolder}/{embedding_name}')

    annotate_intents(data_helper, f'{work_dir}/intents_dictionary.json')

    data_helper.df = data_helper.df[data_helper.df['intent'] != 'others']

    print('Getting without_others_metrics')
    without_others_metrics_df = get_metrics(data_helper)

    all_intents_metrics_df.insert(0, 'intents', 'all_intents')
    without_others_metrics_df.insert(0, 'intents', 'without_others')

    df = pd.concat([all_intents_metrics_df, without_others_metrics_df])

    df.insert(0, 'embedding', embedding_name)

    df.to_csv(f'{work_dir}/clustering_metrics.csv', index=False)

    return df


if __name__ == '__main__':
    for embedding in MAP_EMBEDDING_K_VALUE.keys():
        print(f'Running pipeline for {embedding}')

        run_pipeline(embedding)

from glob import glob

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm.notebook import tqdm

from src.core import file_manager
from src.embeddings.constants import EMBEDDING_MODELS_TRANSLATION
from .chart_helper import *
from .clustering_helper import clustering


def read_file(path):
    return pd.read_json(path, orient='records', dtype={
        'txt': str,
        'embeddings': np.array
    }, lines=True)


def read_multiple_files(paths):
    return pd.concat(Parallel(n_jobs=-1)(delayed(read_file)(i) for i in paths), ignore_index=True)


def make_plot_df(arr, model):
    dict_ = {
        'k': arr[:, 0],
        'silhouette': arr[:, 1],
        'davies_bouldin': arr[:, 2]
    }
    df = pd.DataFrame(dict_)
    df['model'] = model

    return df


def make_trains_apply_metrics(k, embeddings):
    labels, _ = clustering(k, X=embeddings.float())

    return [k, silhouette_score(embeddings, labels, metric='cosine'), davies_bouldin_score(embeddings, labels)]


class MetricHelperBase:
    def __init__(self, actor):
        self.actor = actor

        self.metrics_path = file_manager.filename_from_data_dir(f'embeddings/{self.actor}_metrics.csv')


class MetricHelperGenerator(MetricHelperBase):
    def __init__(self, actor, random_seed=42, k_range=np.arange(2, 101)):
        super().__init__(actor)
        self.random_seed = random_seed
        self.k_range = k_range

    def generate_metrics(self, embedding):
        return Parallel(n_jobs=2)(
            delayed(make_trains_apply_metrics)(i, embedding) for i in tqdm(self.k_range)
        )

    def get_embeddings(self, embedding_model):
        embedding_dir = file_manager.filename_from_data_dir(f'embeddings/{embedding_model}')

        filenames = glob(f'{embedding_dir}/text_emb_{self.actor}.json/*.json')

        df = read_multiple_files(filenames)
        arr = np.array(df['embeddings'].map(lambda x: np.array(x[0])).to_list())

        embeddings = torch.from_numpy(arr)
        return embeddings

    def generate_metrics_for_all_embeddings(self):
        dfs_metrics = []
        for embedding_model in EMBEDDING_MODELS_TRANSLATION.keys():
            print(f'getting {embedding_model} embeddings')
            embeddings = self.get_embeddings(embedding_model)

            metrics = self.generate_metrics(embeddings)

            print('sorting metrics values')
            arr = np.array(sorted(metrics, key=lambda x: x[0]))

            print('generating metrics dataframe')
            dfs_metrics.append(make_plot_df(arr, embedding_model))

        df_merged_metrics = pd.concat(dfs_metrics)

        df_merged_metrics['modelo'] = df_merged_metrics['model'].map(EMBEDDING_MODELS_TRANSLATION)

        df_merged_metrics.to_csv(self.metrics_path, index=False)

        return df_merged_metrics


class MetricHelperPresenter(MetricHelperBase):
    def __init__(self, actor):
        super().__init__(actor)

        self.df = pd.read_csv(self.metrics_path)

    def show_davies_bouldin_score(self):
        plot_charts(
            self.df,
            y_column='davies_bouldin',
            title='',
            color='modelo',
            xaxis_title='K',
            yaxis_title='Davies Boundin'
        )

    def show_silhouette_score(self):
        plot_charts(
            self.df,
            y_column='silhouette',
            title='',
            color='modelo',
            xaxis_title='K',
            yaxis_title='Silhouette Score'
        )

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm.notebook import tqdm

from src.clustering.clustering_helper import ClusteringHelper
from src.core import file_manager as fm
from src.core.chart_helper import *
from src.embeddings.constants import EMBEDDING_MODELS_TRANSLATION


def make_plot_df(arr, model):
    dict_ = {
        'k': arr[:, 0],
        'silhouette': arr[:, 1],
        'davies_bouldin': arr[:, 2]
    }
    df = pd.DataFrame(dict_)
    df['model'] = model

    return df


def get_clustering_metrics(labels, embeddings):
    return silhouette_score(embeddings, labels, metric='cosine'), davies_bouldin_score(embeddings, labels)


def make_trains_apply_metrics(k, embeddings):
    labels = ClusteringHelper(n_clusters=k, embeddings=embeddings).get_labels_as_numpy()

    return [k, *get_clustering_metrics(labels=labels, embeddings=embeddings)]


class MetricHelperBase:
    def __init__(self, actor):
        self.actor = actor

        self.metrics_path = fm.filename_from_data_dir(f'embeddings/{self.actor}_metrics.csv')


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
        df = fm.read_json_of_dir(
            fm.filename_from_data_dir(f'embeddings/{embedding_model}/text_emb_{self.actor}.json')
        )

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
        self.df['Embedding Model'] = self.df['modelo']

    def show_davies_bouldin_score(self):
        plot_line_chart(
            self.df,
            x='k',
            y='davies_bouldin',
            title='Davies Bouldin Scores of all Embedding Models',
            color='Embedding Model',
            xaxis_title='K',
            yaxis_title='Davies Bouldin Score'
        )

    def show_silhouette_score(self):
        plot_line_chart(
            self.df,
            x='k',
            y='silhouette',
            title='Silhouette Scores of all Embedding Models',
            color='Embedding Model',
            xaxis_title='K',
            yaxis_title='Silhouette Score'
        )

import numpy as np
import pandas as pd
from scipy.spatial import distance

from src.core import file_manager as fm
from src.core.distribution_data import get_distribution_data


def get_cosine_distance_or_nan(vector1, vector2):
    if np.all(np.array(vector1) == 0):
        return np.nan
    return distance.cosine(vector1, vector2)


class DataHelper:
    def __init__(self, embedding_name, actor):
        self.embedding_name = embedding_name
        self.actor = actor
        self.df = self.read_annotated_txt()
        self.original_df = self.df.copy(deep=True)

    def read_annotated_txt(self):
        df = fm.read_json_of_dir(
            fm.filename_from_data_dir(f'embeddings/{self.embedding_name}/text_emb_{self.actor}.json'),
            lines=True
        )

        annotated_sentences_df = pd.read_csv(
            fm.filename_from_data_dir(f'output/{self.actor}/annotated_sentences.csv')
        )

        df['annotated_txt'] = annotated_sentences_df['annotated_txt']

        return df

    def reset_df(self):
        self.df = self.original_df

    def get_embeddings(self):
        embeddings = np.array(self.df['embeddings'].map(lambda x: np.array(x[0])).to_list())

        return embeddings

    def generate_distances_from_centroid(self, centroids):
        self.df['distance'] = self.df.apply(
            lambda row: get_cosine_distance_or_nan(row['embeddings'], centroids[row['label']]), axis=1
        )

        return self.df

    def get_unique_labels(self):
        return sorted(self.df.label.unique())

    def get_distance_distribution_by_label(self):
        labels = self.get_unique_labels()

        return [
            get_distribution_data(self.df[self.df['label'] == label]['distance']) for label in labels
        ]

    def remove_empty_embeddings(self):
        self.df = self.df[self.df.apply(lambda row: not np.all(np.array(row['embeddings'][0]) == 0), axis=1)]

    def remove_sentences_by_measure(self, measure):
        distance_distribution_by_label = self.get_distance_distribution_by_label()

        self.df = self.df[self.df.apply(
            lambda row: row['distance'] <= vars(distance_distribution_by_label[row['label']])[measure], axis=1
        )]

        return self.df

    def remove_outlier_sentences(self):
        self.remove_sentences_by_measure('upper_bound')

    def remove_higher_than_median_sentences(self):
        self.remove_sentences_by_measure('med')

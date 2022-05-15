from glob import glob

import numpy as np
import torch

from src.core import file_manager as fm
from .distribution_data import get_distribution_data


class DataHelper:
    def __init__(self, embedding_name, actor, remove_empty_embeddings=False):
        self.embedding_name = embedding_name
        self.actor = actor

        self.initialize_df(remove_empty_embeddings)

    def initialize_df(self, remove_empty_embeddings):
        self.df = fm.read_multiple_files(self.__get_file_data_paths())

        self.original_df = self.df

        if remove_empty_embeddings:
            self.df = self.df[self.df.apply(lambda row: not np.all(np.array(row['embeddings'][0]) == 0), axis=1)]

    def get_labels(self):
        return sorted(self.df.label.unique())

    def get_distance_distribution_by_label(self, df_to_calculate):
        labels = self.get_labels()

        return [
            get_distribution_data(df_to_calculate[df_to_calculate['label'] == label]['distance']) for label in labels
        ]

    def get_distance_actual_distribution_by_label(self):
        return self.get_distance_distribution_by_label(self.df)

    def get_distance_original_distribution_by_label(self):
        return self.get_distance_distribution_by_label(self.original_df)

    def remove_outlier_sentences(self):
        distance_distribution_by_label = self.get_distance_actual_distribution_by_label()

        self.df = self.df[self.df.apply(
            lambda row: row['distance'] <= distance_distribution_by_label[row['label']].upper_bound, axis=1
        )]

    def remove_higer_than_median_sentences(self):
        distance_distribution_by_label = self.get_distance_actual_distribution_by_label()

        self.df = self.df[self.df.apply(
            lambda row: row['distance'] <= distance_distribution_by_label[row['label']].med, axis=1
        )]

    @staticmethod
    def generate_filename_from_data_dir(filename):
        return fm.filename_from_data_dir(f'{filename}')

    @staticmethod
    def get_embeddings_as_torch(df):
        embeddings_np = np.array(df['embeddings'].map(lambda x: np.array(x[0])).to_list())

        embeddings_torch = torch.from_numpy(embeddings_np)

        return embeddings_torch

    def __get_file_data_paths(self, ):
        paths = glob(fm.filename_from_data_dir(f'embeddings/{self.embedding_name}/text_emb_{self.actor}.json/*.json'))

        return paths

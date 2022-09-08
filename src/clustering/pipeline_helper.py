from collections import defaultdict
from pathlib import Path

from src.clustering.clustering_helper import ClusteringHelper
from src.clustering.data_helper import DataHelper
from src.clustering.tsne_helper import TsneHelper
from src.clustering.wordcloud_helper import print_word_clouds_of_each_label
from src.core import file_manager as fm
from src.core.chart_helper import plot_distance_charts


class PipelineHelper:
    def __init__(self, embedding_name, actor, k, sub_folder_k=None):
        self.annotated_df = None
        self.clustering_helper = None
        self.actor = actor
        self.embedding_name = embedding_name
        self.k = k
        self.sub_folder_k = sub_folder_k

        self.data_helper = DataHelper(self.embedding_name, self.actor)

    def run_clustering(self):
        self.clustering_helper = ClusteringHelper(self.k, self.data_helper.get_embeddings(), self.actor)

        self.data_helper.df['label'] = self.clustering_helper.get_labels_as_numpy()

        self.data_helper.generate_distances_from_centroid(self.clustering_helper.get_centroids_as_numpy())

        self.data_helper.sync_dataframes()

    def ignore_intent(self, intent_name):
        self.data_helper.df = self.data_helper.df[self.annotated_df['intent'] != intent_name]
        self.data_helper.sync_dataframes()
        self.data_helper.reset_df()

    def describe_intents(self, dict_intents):
        df = self.data_helper.df
        index_intents = defaultdict(list)

        print(f'The total of sentences is: {df.txt.count()}')

        for index, intent in dict_intents.items():
            index_intents[intent].append(index)

        for intent in index_intents:
            intent_clusters = index_intents[intent]
            intent_sentences = df[df['label'].isin(index_intents[intent])]

            print(f'{intent}, has {len(intent_clusters)} clusters, and {len(intent_sentences)} sentences')

    def visualize_distance_distribution(self):
        plot_distance_charts(self.data_helper.df)

    def visualize_word_clouds(self, num_sentences=20):
        print_word_clouds_of_each_label(self.data_helper.df, self.data_helper.get_unique_labels(), num_sentences)

    def visualize_tsne(self):
        title_tsne = f'T-SNE algorithm: fast_kmeans model: {self.embedding_name} actor: {self.actor}'

        tsne_helper = TsneHelper(self.data_helper, title_tsne)

        fig = tsne_helper.build_tsne_chart()

        fig.show()


    def save_data(self, df_data, internal_dir=''):
        output_dir = Path(fm.filename_from_data_dir(
            f'output/patient/{internal_dir}{self.sub_folder_k}/{self.embedding_name}')
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = f'{output_dir}/annotated_sentences.csv'

        print(f'saving data at: {output_file}')
        df_data.to_csv(output_file, index=False)

    def annotate_data(self, dict_intents):
        print('applying map intents.....')
        self.data_helper.df['intent'] = self.data_helper.df.loc[:, 'label'].map(dict_intents)
        self.data_helper.original_df['intent'] = self.data_helper.df.loc[:, 'intent']

        self.annotated_df = self.data_helper.df.loc[:, ['txt', 'annotated_txt', 'label', 'distance', 'intent']]

        self.save_data(self.annotated_df)

        without_others_df = self.annotated_df.loc[self.annotated_df['intent'] != 'others']        
        self.save_data(without_others_df, internal_dir='without_others_intent/')

        print('Describing data...')
        self.describe_intents(dict_intents)

        self.annotated_df.head(2)

    def annotate_data_without_outliers(self, sub_folder_k, dict_intents):
        self.sub_folder_k = sub_folder_k

        self.data_helper.reset_df()

        self.data_helper.remove_outlier_sentences()

        self.annotate_data(dict_intents)

    def annotate_data_without_sentences_higher_than_median(self, sub_folder_k, dict_intents):
        self.sub_folder_k = sub_folder_k

        self.data_helper.reset_df()

        self.data_helper.remove_higher_than_median_sentences()

        self.annotate_data(dict_intents)

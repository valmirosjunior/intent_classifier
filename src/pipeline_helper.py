from pathlib import Path

from .chart_helper import plot_distance_charts
from .clustering_helper import ClusteringHelper
from .data_helper import DataHelper
from .tsne_helper import TsneHelper
from .wordcloud_helper import print_word_clouds_of_each_label


class PipelineHelper:

    def __init__(self, embedding_name, actor, n_clusters, sub_folder_k=None,  **kwargs):
        self.actor = actor
        self.embedding_name = embedding_name
        self.n_clusters = n_clusters
        self.sub_folder_k = sub_folder_k

        self.initialize_data(**kwargs)

    def initialize_data(self,  **kwargs):
        remove_empty_embeddings = kwargs.get('remove_empty_embeddings', True)
        remove_outliers = kwargs.get('remove_outliers', True)
        remove_higer_than_median = kwargs.get('remove_higer_than_median', False)

        self.data_helper = DataHelper(self.embedding_name, self.actor, remove_empty_embeddings)
        self.clustering_helper = ClusteringHelper(self.n_clusters, self.embedding_name, self.actor, self.data_helper)

        self.data_helper.df['label'] = self.clustering_helper.get_labels()

        self.data_helper.df['distance'] = self.clustering_helper.get_distance_from_centroid()

        self.data_helper.original_df = self.data_helper.df

        if remove_outliers:
            self.data_helper.remove_outlier_sentences()

        elif remove_higer_than_median:
            self.data_helper.remove_higer_than_median_sentences()


    def visualize_distance_distribution(self):
        plot_distance_charts(self.data_helper.original_df)

    def visualize_word_clouds(self, num_sentences=20):
        print_word_clouds_of_each_label(self.data_helper, num_sentences)

    def visualize_tsne(self):
        title_tsne = f'T-SNE algorithm: fast_kmeans model: {self.embedding_name} actor: {self.actor}'

        tsne_helper = TsneHelper(self.data_helper, title_tsne)

        fig = tsne_helper.build_tsne_chart()

        fig.show()

    def annotate_data(self, dict_intents):
        self.annotated_df = self.data_helper.df

        print('applying map intents.....')

        self.annotated_df['intent'] = self.annotated_df['label'].map(dict_intents)

        count_before = self.annotated_df['txt'].count()

        print(f'Before dropping nan values: {count_before}')
        print('dropping nan intents.....')
        # remove nan intents
        self.annotated_df = self.annotated_df.dropna()

        count_after = self.annotated_df['txt'].count()
        
        print(f'After dropping nan values: {count_after}')
        print(f'The total of data droped was {count_before - count_after} rows')

        print('dropping unnecessary columns.....')
        self.annotated_df = self.annotated_df.drop(['embeddings'], axis=1)

        output_dir = Path(f'{DataHelper.DATA_DIR}/output/patient/{self.sub_folder_k}/{self.embedding_name}')

        output_dir.mkdir(parents=True, exist_ok=True)

        print('saving data')

        self.annotated_df.to_csv(f'{output_dir}/classified_sentences.csv', index=False)

        self.annotated_df.head(2)

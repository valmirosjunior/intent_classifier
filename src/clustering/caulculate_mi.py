import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.clustering.pipeline_helper_retriever import *
from src.core import file_manager as fm


class CalculatorMutualInformation:
    pipeline_helper_retriever: PipelineHelperRetriever

    def __init__(self, embedding_name, use_intent_index=False, actor='patient', sub_folder_k='k100'):
        self.embedding_name = embedding_name
        self.use_intent_index = use_intent_index
        self.actor = actor
        self.sub_folder_k = sub_folder_k

    def get_data_for_calculate(self):
        data_helper = self.pipeline_helper_retriever.pipeline_helper.data_helper

        if self.use_intent_index:
            labels = data_helper.df['intent'].map(MAP_INTENT_INDEX).to_numpy()
        else:
            labels = data_helper.df.label.to_numpy()

        return data_helper.get_embeddings(), labels

    def get_mi(self, n_neighbor=3):
        embeddings, labels = self.get_data_for_calculate()

        return mutual_info_classif(embeddings, labels, n_neighbors=n_neighbor)

    def get_mi_for_variations(self):
        # pipeline_helper_retriever, use_intent_index
        data_helper = self.pipeline_helper_retriever.pipeline_helper.data_helper
        metrics_all = self.get_mi()

        data_helper.remove_outlier_sentences()
        metrics_not_outliers = self.get_mi()

        data_helper.remove_higher_than_median_sentences()
        metrics_not_median = self.get_mi()

        data = {
            'variation': ['all', 'not_outliers', 'not_median'],
            'mutual_information': [
                [metrics_all.tolist()],
                [metrics_not_outliers.tolist()],
                [metrics_not_median.tolist()]
            ],
        }

        return pd.DataFrame(data)

    def run_pipeline(self):
        self.pipeline_helper_retriever = PipelineHelperRetriever(
            embedding_name=self.embedding_name, actor=self.actor, sub_folder_k=self.sub_folder_k
        )

        print('Getting all_intents mutual information')
        all_intents_mutual_information_df = self.get_mi_for_variations()

        print('Getting without_others mutual information')
        self.pipeline_helper_retriever.pipeline_helper.data_helper.reset_df()
        self.pipeline_helper_retriever.pipeline_helper.annotate_data(self.pipeline_helper_retriever.dict_intents)
        self.pipeline_helper_retriever.pipeline_helper.ignore_intent('others')

        without_others_mutual_information_df = self.get_mi_for_variations()

        all_intents_mutual_information_df.insert(0, 'intents', 'all_intents')
        without_others_mutual_information_df.insert(0, 'intents', 'without_others')

        df = pd.concat([all_intents_mutual_information_df, without_others_mutual_information_df])
        df.insert(0, 'embedding', self.embedding_name)

        prefix_name = 'grouped_by_intent_' if self.use_intent_index else ''

        output_file = fm.filename_from_data_dir(
            f'output/{self.actor}/{self.sub_folder_k}/{self.embedding_name}/{prefix_name}mutual_information.csv'
        )

        df.to_csv(output_file, index=False)

        print(f'The results were saved at: {output_file}')

        return df

    def calculate_mi_for_range(self, range_n_neighbors):
        dfs = []
        for n_neighbor in range_n_neighbors:
            print(f'calculating mi for {self.embedding_name} with n_neighbors={n_neighbor}')
            mutual_information = self.get_mi(n_neighbor)

            dfs.append(pd.DataFrame({
                'n_neighbor': [int(n_neighbor)] * len(mutual_information),
                'feature': range(len(mutual_information)),
                'mutual_information': mutual_information,
            }))

        df_merged = pd.concat(dfs)
        df_merged.insert(0, 'embedding', self.embedding_name)

        return df_merged

    def run_pipeline_for_range_n_neighbors(self):
        self.pipeline_helper_retriever = PipelineHelperRetriever(
            embedding_name=self.embedding_name, actor=self.actor, sub_folder_k=self.sub_folder_k
        )

        self.pipeline_helper_retriever.pipeline_helper.annotate_data_without_sentences_higher_than_median(
            sub_folder_k='k100_without_sentences_higher_than_median/',
            dict_intents=self.pipeline_helper_retriever.dict_intents
        )

        self.pipeline_helper_retriever.pipeline_helper.ignore_intent('others')

        df = self.calculate_mi_for_range(range(3, 101))

        output_file = fm.filename_from_data_dir(
            f'output/{self.actor}/{self.sub_folder_k}/{self.embedding_name}/mutual_information_by_range_n_neighbors.csv'
        )

        df.to_csv(output_file, index=False)

        print(f'The results were saved at: {output_file}')

        return df


if __name__ == '__main__':
    for embedding in MAP_EMBEDDING_K_VALUE.keys():
        print(f'Running pipeline for {embedding}')

        # CalculatorMutualInformation(embedding_name=embedding, use_intent_index=False).run_pipeline()
        # CalculatorMutualInformation(embedding_name=embedding, use_intent_index=True).run_pipeline()

        CalculatorMutualInformation(embedding_name=embedding).run_pipeline_for_range_n_neighbors()

        # run_pipeline(embedding, use_intent_index=False)
        # run_pipeline(embedding, use_intent_index=True)

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.clustering.pipeline_helper_retriever import *
from src.core import file_manager as fm


# def get_mi(embeddings, labels):
#     return mutual_info_classif(embeddings, labels)
#     # return mutual_info_classif(data_helper.get_embeddings(), data_helper.df.label.to_numpy())


class CalculatorMutualInformation:
    pipeline_helper_retriever: PipelineHelperRetriever

    def __init__(self, embedding_name, use_intent_index, actor='patient', sub_folder_k='k100'):
        self.embedding_name = embedding_name
        self.use_intent_index = use_intent_index
        self.actor = actor
        self.sub_folder_k = sub_folder_k

    def get_data_for_calculate(self):
        data_helper = self.pipeline_helper_retriever.pipeline_helper.data_helper

        if self.use_intent_index:
            labels = data_helper.df['intent'].map(MAP_INTENT_INDEX).to_numpy()
            print(f'using this intent_index: {labels}')
        else:
            labels = data_helper.df.label.to_numpy()
            print(f'using this labels: {labels}')

        return data_helper.get_embeddings(), labels

    def get_mi(self):
        embeddings, labels = self.get_data_for_calculate()

        return mutual_info_classif(embeddings, labels)

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


if __name__ == '__main__':
    for embedding in MAP_EMBEDDING_K_VALUE.keys():
        print(f'Running pipeline for {embedding}')

        # CalculatorMutualInformation(embedding_name=embedding, use_intent_index=False).run_pipeline()
        CalculatorMutualInformation(embedding_name=embedding, use_intent_index=True).run_pipeline()

        # run_pipeline(embedding, use_intent_index=False)
        # run_pipeline(embedding, use_intent_index=True)

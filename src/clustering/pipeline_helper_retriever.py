import json

from src.clustering.pipeline_helper import PipelineHelper
from src.core import file_manager as fm

MAP_EMBEDDING_K_VALUE = {
    'bert_pt': 91,
    'flair_pt': 95,
    'glove': 99,
    'lasbe': 81,
    'use': 82,
}

MAP_INTENT_INDEX = {
    'greeting': 0,
    'inform_medicine': 1,
    'inform_symptoms': 2,
    'others': 3,
    'request_inform': 4
}


class PipelineHelperRetriever:
    def __init__(self, embedding_name, actor, sub_folder_k):
        self.embedding_name = embedding_name
        self.actor = actor
        self.sub_folder_k = sub_folder_k
        self.dict_intents = self.load_dict_intent()
        self.pipeline_helper = self.load_pipeline_from_saved_dict_intent()

    def get_work_dir(self):
        return fm.filename_from_data_dir(
            f'output/{self.actor}/{self.sub_folder_k}/{self.embedding_name}'
        )

    def load_dict_intent(self):
        intents_dictionary_file = open(f'{self.get_work_dir()}/intents_dictionary.json')

        return {
            int(key): value
            for key, value in json.load(intents_dictionary_file).items()
        }

    def load_pipeline_from_saved_dict_intent(self):
        self.pipeline_helper = PipelineHelper(
            embedding_name=self.embedding_name,
            actor=self.actor,
            k=MAP_EMBEDDING_K_VALUE[self.embedding_name],
            sub_folder_k=self.sub_folder_k
        )

        print('Run clustering')
        self.pipeline_helper.run_clustering()

        print('Annotate data...')
        self.pipeline_helper.annotate_data(self.dict_intents)

        return self.pipeline_helper

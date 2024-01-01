import pandas as pd
import os
import tensorflow_hub

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from pathlib import Path


from src.core import file_manager as fm

# bert:
#   neuralmind/bert-base-portuguese-cased

# flair:
#   pt-forward
#   pt-backward

# glove:
#   sentence-transformers/average_word_embeddings_glove.6B.300d

# labse:
#   sentence-transformers/LaBSE

# muse:
#   distiluse-base-multilingual-cased
#   tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


class BERTopicHelper:
    def __init__(self, model_name, docs):
        self.model_name = model_name
        self.docs = docs


    def run_pipeline(self):
        print("Creating and fiting, the model...")
        ber_topic_model = self.build_model()

        doc_info = ber_topic_model.get_document_info(self.docs)

        print("Adding labels and txt column...")
        doc_info['label'] = doc_info['Topic']
        doc_info['txt'] = doc_info['Document']


        # print("Saving data to file")
        # output_dir = Path(fm.filename_from_data_dir(f'output/patient/bertopic/{self.model_name}'))
        # output_dir.mkdir(parents=True, exist_ok=True)
        # output_file = f'{output_dir}/annotated_sentences.csv'

        # doc_info.to_csv(output_file, index=False)
        
        self.doc_info = doc_info
        return ber_topic_model, doc_info

    
    def build_model(self):
        if self.model_name == "bert_pt":
            return self.build_bert_model()
        elif self.model_name == "flair_pt":
            return self.build_flair_model()
        elif self.model_name == "glove":
            return self.build_glove_model()
        elif self.model_name == "labse":
            return self.build_labse_model()
        elif self.model_name == "use":
            return self.build_muse_model()
        else:
            raise Exception("Woops, wrong model_name...")


    def build_bert_model(self):
        embedding_model = pipeline("feature-extraction", model="neuralmind/bert-base-portuguese-cased")
        bert_model = BERTopic(embedding_model=embedding_model)

        bert_model.fit_transform(self.docs)

        return bert_model
    
    
    def build_glove_model(self):
        glove_embedding_model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
        glove_model = BERTopic(language="multilingual", embedding_model=glove_embedding_model)

        glove_model.fit_transform(self.docs)

        return glove_model
    

    def build_flair_model(self):
        stacked_embeddings = StackedEmbeddings([FlairEmbeddings('pt-forward'), FlairEmbeddings('pt-backward')])

        flair_model = BERTopic(embedding_model=stacked_embeddings)
        flair_model.fit_transform(self.docs)
        
        return flair_model
    
    
    def build_labse_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"]=""

        labse_embedding_model = SentenceTransformer('sentence-transformers/LaBSE')
        labse_model = BERTopic(language="multilingual", embedding_model=labse_embedding_model)
        labse_model.fit_transform(self.docs)

        del os.environ['CUDA_VISIBLE_DEVICES']
        
        return labse_model
    

    def build_muse_model(self):
        embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        muse_model = BERTopic(embedding_model=embedding_model)

        muse_model.fit_transform(self.docs)
        
        return muse_model
    

    def build_muse_model_with_sentence_transformer(self):
        model = SentenceTransformer('distiluse-base-multilingual-cased')
        muse_model = BERTopic(language="multilingual", embedding_model=model)
        muse_model.fit_transform(self.docs)
        
        return muse_model


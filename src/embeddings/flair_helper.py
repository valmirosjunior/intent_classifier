import pandas as pd
from flair.embeddings import FlairEmbeddings, DocumentPoolEmbeddings
from flair.data import Sentence
from pathlib import Path

from src.core import file_manager


def get_document_embedding():
    flair_embedding_forward = FlairEmbeddings('pt-forward')
    flair_embedding_backward = FlairEmbeddings('pt-backward')

    return DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward])


class FlairHelper:
    def __init__(self, actor):
        self.actor = actor

        self.path_annotated_sentences = file_manager.filename_from_data_dir(
            f'output/{self.actor}/annotated_sentences.csv'
        )

        self.annotated_sentences = pd.read_csv(self.path_annotated_sentences)

        self.document_embeddings = get_document_embedding()

    def sentence_embedding(self, txt):
        sentence = Sentence(str(txt))

        self.document_embeddings.embed(sentence)

        return [sentence.embedding.tolist()]

    def generate_embedding_flair_pt(self):
        print('generating embeddings...')
        self.annotated_sentences['embeddings'] = self.annotated_sentences['txt'].apply(self.sentence_embedding)

        print('saving embeddings...')

        output_dir = Path(file_manager.filename_from_data_dir(f'embeddings/flair_pt/text_emb_{self.actor}.json'))

        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = f'{output_dir}/text_emb_{self.actor}.json'

        print('saving data....')
        self.annotated_sentences.to_json(output_file, orient="records", lines=True)

        return self.annotated_sentences

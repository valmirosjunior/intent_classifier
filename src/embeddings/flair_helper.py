import math
import pandas as pd
from flair.embeddings import DocumentPoolEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence
from pathlib import Path

from src.core import file_manager as fm



flair_embedding_forward = FlairEmbeddings('pt-forward')
flair_embedding_backward = FlairEmbeddings('pt-backward')
document_pool_embeddings = DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward])
stacked_embeddings = StackedEmbeddings([flair_embedding_forward, flair_embedding_backward])


def get_sentence_embedding(txt):  
  sentence = Sentence(str(txt))
  
  document_pool_embeddings.embed(sentence)
  stacked_embeddings.embed(sentence)

  data = {
    'tokens': [token.text for token in sentence.tokens],
    # 'word_embeddings': [token.embedding.tolist() for token in sentence.tokens],
    'embeddings': [sentence.embedding.tolist()]
  }

  return data


class FlairHelper:
    def __init__(self, actor):
        self.actor = actor

        self.path_annotated_sentences = fm.filename_from_data_dir(
            f'output/{self.actor}/annotated_sentences.csv'
        )

        self.annotated_sentences = pd.read_csv(self.path_annotated_sentences)


    def generate_embedding_flair_pt(self, interval=5000):
      output_dir = fm.create_dir_if_not_exists_on_data_dir(f'embeddings/flair_pt/text_emb_{self.actor}.json')

      # for offset in range(math.ceil(self.annotated_sentences.txt.count()/interval)):
      #   start = offset * interval
      #   end = start + interval
      #   print(f'generating embeddings to the interval: {start} --> {end}')
        
      #   df_interval = self.annotated_sentences[self.annotated_sentences.index.isin(range(start, end))]

      #   data = map(get_sentence_embedding, df_interval['txt'].to_numpy())

      #   df_data  = pd.DataFrame(list(data))

      #   df_embeddings = pd.concat([df_interval, df_data.set_index(df_interval.index)], axis=1, ignore_index=True)

      #   print('saving data....')
      #   df_embeddings.to_json( f'{output_dir}/part_{offset}_text_embeddings.json', orient="records", lines=True)


      print('generating embeddings...')
      data = map(get_sentence_embedding, self.annotated_sentences['txt'].to_numpy())

      df_data  = pd.DataFrame(list(data))

      df_embeddings = pd.concat([self.annotated_sentences[['txt']], df_data], axis=1)

      print('saving data....')
      df_embeddings.to_json( f'{output_dir}/text_emb_patient.json', orient="records", lines=True)

        


if __name__ == '__main__':
    flair_helper = FlairHelper(actor='patient')

    flair_helper.generate_embedding_flair_pt()
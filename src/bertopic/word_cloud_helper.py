import json
import pandas as pd

from collections import defaultdict
from pathlib import Path

from src.clustering import wordcloud_helper
from src.core import file_manager as fm


class WordCloudHelper:
  def __init__(self, model_name):
        self.model_name = model_name
        self.work_dir = fm.filename_from_data_dir(f'output/patient/bertopic/{self.model_name}')
        self.setup()


  def setup(self):
    doc_info_file = f'{self.work_dir}/annotated_sentences.csv'
    self.doc_info = pd.read_csv(doc_info_file)
    
    self.labels = self.doc_info.label.unique()
    self.labels.sort()



  def generate_word_cloud(self, n_segment, segment_size = 50):    
    start = n_segment * segment_size
    end = start + segment_size
    labels_segment = self.labels[start:end]

    return wordcloud_helper.print_word_clouds_of_each_label(self.doc_info, labels_segment)

  
  def dict_intents(self):    
    intents_dictionary = open(f'{self.work_dir}/intents_dictionary.json')

    return {
      int(key): value
      for key, value in json.load(intents_dictionary).items()
    }

  def anottate_intents(self):
    self.doc_info['intent'] = self.doc_info['label'].map(self.dict_intents())
    self.doc_info.to_csv((f'{self.work_dir}/annotated_sentences.csv'))
    self.doc_info.head(2)


  def check_intents(self):
    ### Check Intents
    intents = self.doc_info.intent.unique()

    print('intents:', intents)
    keys = list(self.dict_intents().keys())

    for i in range(-1, len(keys) -1):
      if not i in keys:
        print(i, 'is missing...')

    print('last indnex:', i)
  

  # def count_intents(self):
  #   ### Check Intents
  #   intents = intents_dictionary = open(f'{self.work_dir}/intents_dictionary.json')

  #   return intents

  def describe_intents(self):
    df = self.doc_info
    index_intents = defaultdict(list)

    print(f'The total of sentences is: {df.txt.count()}')

    intents_dictionary = open(f'{self.work_dir}/intents_dictionary.json')
    dict_intents = json.load(intents_dictionary)

    print(f'The total of topics is: {len(dict_intents)}')

    for index, intent in dict_intents.items():
        index_intents[intent].append(int(index))

    for intent in index_intents:
        intent_topics = index_intents[intent]
        intent_sentences = df[df['label'].isin(index_intents[intent])]

        print(f'{intent}, has {len(intent_topics)} topics, and {len(intent_sentences)} sentences')


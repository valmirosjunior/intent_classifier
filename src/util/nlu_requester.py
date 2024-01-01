import requests
import pandas as pd
import src.core.file_manager as fm

import argparse

base_dir = fm.filename_from_data_dir(
  'output/patient/without_others_intent/k100_without_sentences_higher_than_median'
)

intent_map = {
  'greeting': 0,
  'inform_medicine': 1,
  'inform_symptoms': 2,
  'request_inform': 3
}

def read_args():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-m', '--model',
                       help='The model that of the nlu server that is running',
                       required=True)
  # --no-validation sets it to False
  parser.add_argument('-v', '--validation',
                       help='The flag to indicates if it is the testing o validation data',
                       action=argparse.BooleanOptionalAction,
                       required=True)

  return parser.parse_args()  



if __name__ == '__main__':
  embedding_model = read_args().model
  is_validation = read_args().validation
  dataset = "Validation" if is_validation else "Testing"

  print('working with the model:', embedding_model)
  print(f'working with {dataset} dataset', "=>", is_validation)

  output_directory = f'{base_dir}/{embedding_model}'

  if is_validation:
    input_file = f'{base_dir}/intersection_300_sentences_with_label.csv'
    file_to_save = f'{output_directory}/nlu_predictions_of_intersection.csv'
  else:
    input_file = f'{output_directory}/test_data.csv'
    file_to_save = f'{output_directory}/nlu_predictions.csv'

  print('Reading csv...')
  df = pd.read_csv(input_file)

  print('Starting predictions...')
  predicitions = []
  for index, row in df.iterrows():
    prediciton = requests.post('http://localhost:5005/model/parse', json={'text': row['txt']})
    
    predicitions.append(prediciton.json()['intent']['name'])
  print('Predictions done....')
  
  df_to_save = pd.DataFrame({'original_intent': df['intent'], 'intent_predicted': predicitions })
  
  print(f'saving file at: {file_to_save}')
  df_to_save.to_csv(file_to_save)

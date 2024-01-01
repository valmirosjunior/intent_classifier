import pandas as pd

from pathlib import Path


import src.core.file_manager as fm

from src.core.chart_helper import plot_distribution_charts
from src.core.distribution_data import get_distribution_data


class DataHelper:
    def __init__(self, embedding_name, actor='patient'):
        self.embedding_name = embedding_name
        self.actor = actor
        self.work_dir = fm.filename_from_data_dir(f'output/patient/bertopic/{embedding_name}')
        
        df = pd.read_csv(f'{self.work_dir}/annotated_sentences.csv')
        self.df_data = df[~df['intent'].isin(['outliers'])]

    
    def plot_probability_distribution_charts(self):
        plot_distribution_charts(
          self.df_data,
          distribution_collumn="Probability",
          short_title="Probabilidade",
          long_title="distribiuição das probabilidades"
        )


    def remove_sentences_by_measure(self, measure):
        labels = sorted(self.df_data.label.unique())

        probabilities_distribution_by_label = {
            label: get_distribution_data(self.df_data[self.df_data['label'] == label]['Probability']) for label in labels
        }

        result_df = self.df_data[self.df_data.apply(
            lambda row: row['Probability'] >= vars(probabilities_distribution_by_label[row['label']])[measure], axis=1
        )]

        return result_df
    
    
    def save_variation(self, variation, measure):
      df = self.remove_sentences_by_measure(measure)

      output_dir = Path(f'{self.work_dir}/{variation}')
      output_dir.mkdir(parents=True, exist_ok=True)

      df.to_csv(f'{output_dir}/annotated_sentences.csv', index=False)
      
      return df


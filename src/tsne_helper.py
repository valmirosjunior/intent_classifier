import pandas as pd
import plotly.express as px
from MulticoreTSNE import MulticoreTSNE as TSNE

from .data_helper import DataHelper


class TsneHelper:

    def __init__(self, data_helper, title):
        self.data_helper = data_helper
        self.title = title

    def build_tsne_chart(self):
        df_plot = self.__build_df_chart_data()

        fig = px.scatter(df_plot, x="x", y="y", color="label", hover_data=['text', 'label'], title=self.title)

        return fig

    def __build_df_chart_data(self):
        print('Transforming embeddings')

        df_sorted = self.data_helper.df.sort_values(by=['label'])

        transformed_embedded = TSNE(n_jobs=-1).fit_transform(DataHelper.get_embeddings_as_torch(df_sorted))

        print('Creating chart....')
        labels = df_sorted['label']

        dict_ = {
            'x': transformed_embedded[:, 0],
            'y': transformed_embedded[:, 1],
            'label': list(map(lambda x: f'Cluster número {x} | {len(labels[labels == x])} frases', labels)),
            'text': df_sorted['txt'].to_list()
        }

        return pd.DataFrame(dict_)

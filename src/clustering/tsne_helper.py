import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE


class TsneHelper:

    def __init__(self, data_helper, title):
        self.data_helper = data_helper
        self.title = title

    def build_tsne_chart(self):
        df_plot = self.__build_df_chart_data()

        # fig = px.scatter(df_plot, x="x", y="y", color="label", hover_data=['text', 'label'], title=self.title)
        
        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            color="cluster",
            color_continuous_scale='Bluered_r',
            hover_data=['text', 'cluster']
        )

        fig.update_layout(
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(size=18)
        )

        return fig, df_plot

    def __build_df_chart_data(self):
        print('Transforming embeddings')

        df_sorted = self.data_helper.df.sort_values(by=['label'])
        embeddings = np.array(df_sorted['embeddings'].map(lambda x: np.array(x[0])).to_list())

        transformed_embedded = TSNE(n_jobs=-1).fit_transform(torch.from_numpy(embeddings))

        print('Creating chart....')
        labels = df_sorted['label']

        dict_ = {
            'x': transformed_embedded[:, 0],
            'y': transformed_embedded[:, 1],
            'cluster': list(map(lambda x: str(x).zfill(2), labels)),
            # 'label': list(map(lambda x: f'Cluster {x} with {len(labels[labels == x])} Sentences', labels)),
            'text': df_sorted['txt'].to_list()
        }

        return pd.DataFrame(dict_)

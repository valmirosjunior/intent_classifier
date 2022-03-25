import pandas as pd

from .chart_helper import *
from .data_helper import DataHelper


class MetricHelper:
    def __init__(self, actor):
        self.df = pd.read_csv(f'{DataHelper.DATA_DIR}/models/results_merged_{actor}.csv')

    def show_davies_bouldin_score(self):
        plot_charts(
            self.df,
            y_column='davies bouldin',
            title='',
            color='modelo',
            xaxis_title='K',
            yaxis_title='Davies Boundin'
        )

    def show_silhouette_score(self):
        plot_charts(
            self.df,
            y_column='scores',
            title='',
            color='modelo',
            xaxis_title='K',
            yaxis_title='Silhouette Score'
        )

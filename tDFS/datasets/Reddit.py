import os
import pandas as pd
import numpy as np
import wget

from tDFS.datasets.Abstract import AbstractDataModule
from tDFS.constants import DATA_PATH


class RedditDataModule(AbstractDataModule):
    """
    DataModule handling data for the reddit dataset.
    Please find the datasets in: https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def get_data(self):
        file_path = os.path.join(DATA_PATH, f'{self.dataset}.tsv')
        if not os.path.exists(file_path):
            wget.download(f'https://snap.stanford.edu/data/{self.dataset}.tsv', file_path)
        df = pd.read_csv(file_path, delimiter='\t', parse_dates=['TIMESTAMP'])

        df = df.rename({'SOURCE_SUBREDDIT': 'source', 'TARGET_SUBREDDIT': 'target', 'TIMESTAMP': 'timestamp', 'LINK_SENTIMENT': 'label', 'PROPERTIES': 'features'}, axis=1)
        df['features'] = df['features'].map(lambda features: np.array(eval(features)))
        df['timestamp'] = df['timestamp'].map(lambda ts: ts.timestamp())
        df = df[['source', 'target', 'timestamp', 'label', 'features']]

        df, column2map = self.quantile_and_factorize(df, factorize_columns=['source', 'target'], merge=True)

        return df, None, column2map

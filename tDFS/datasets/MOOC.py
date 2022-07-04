import os
import pandas as pd
import numpy as np
import wget
import tarfile

from tDFS.datasets.Abstract import AbstractDataModule
from tDFS.constants import DATA_PATH


class MOOCDataModule(AbstractDataModule):
    """
    DataModule handling data for the act-mooc dataset.
    Please find the datasets in: https://snap.stanford.edu/data/act-mooc.html
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def get_data(self):
        file_path = os.path.join(DATA_PATH, f'{self.dataset}.tar.gz')
        if not os.path.exists(file_path):
            wget.download(f'https://snap.stanford.edu/data/{self.dataset}.tar.gz', file_path)
            tarfile.open(file_path).extractall(DATA_PATH)
            os.remove(file_path)
        actions_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'mooc_actions.tsv'), delimiter='\t', parse_dates=['TIMESTAMP'])
        actions_df = actions_df.rename({'USERID': 'source', 'TARGETID': 'target', 'TIMESTAMP': 'timestamp'}, axis=1)
        actions_df['timestamp'] = actions_df['timestamp'].astype(float)
        features_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'mooc_action_features.tsv'), delimiter='\t')
        features_df['features'] = features_df.apply(lambda row: np.array(row.values[1:]), axis=1)
        features_df = features_df[['ACTIONID', 'features']]
        labels_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'mooc_action_labels.tsv'), delimiter='\t')
        labels_df = labels_df.rename({'LABEL': 'label'}, axis=1)
        df = actions_df.merge(features_df, on='ACTIONID')
        df = df.merge(labels_df, on='ACTIONID')

        df = df[['source', 'target', 'timestamp', 'label', 'features']]

        df, column2map = self.quantile_and_factorize(df, factorize_columns=['source', 'target'])

        return df, None, column2map

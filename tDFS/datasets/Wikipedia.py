import os
import pandas as pd
import numpy as np
import wget

from tDFS.datasets.Abstract import AbstractDataModule
from tDFS.constants import DATA_PATH


class WikipediaDataModule(AbstractDataModule):
    """
    DataModule handling data for the wikipedia datasets.
    Please find the datasets in: http://snap.stanford.edu/jodie/wikipedia.csv
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def get_data(self):
        file_path = os.path.join(DATA_PATH, f'{self.dataset}.csv')
        if not os.path.exists(file_path):
            wget.download(f'http://snap.stanford.edu/jodie/{self.dataset}.csv', file_path)
        df = pd.read_csv(file_path, skiprows=1, header=None)

        df = df.rename({0: 'source', 1: 'target', 2: 'timestamp', 3: 'label'}, axis=1)
        df['features'] = df.apply(lambda row: np.array(row.values[4:]), axis=1)
        df = df[['source', 'target', 'timestamp', 'label', 'features']]

        df, column2map = self.quantile_and_factorize(df, factorize_columns=['source', 'target'])

        return df, None, column2map

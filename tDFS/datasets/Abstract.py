import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from abc import ABC, abstractmethod

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from tDFS.constants import CACHE_PATH
from tDFS.utils.graph import NeighborFinder, RandEdgeSampler

from tDFS.utils.general import load_object, save_object


class AbstractDataModule(pl.LightningDataModule, ABC):
    """
    Abstract DataModule.
    """
    def __init__(self, uniform, batch_size=32, num_workers=0):
        super().__init__()
        self.uniform = uniform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None

    def prepare_data(self):
        dataset_path = os.path.join(CACHE_PATH, f"{self.dataset}.pkl")
        if os.path.exists(dataset_path):
            self.df, self.edge_features, self.node_features, self.column2map = load_object(dataset_path)
            return

        self.df, self.node_df, self.column2map = self.get_data()
        if self.node_df is None:
            self.node_df = pd.DataFrame(np.unique(self.df['source'].tolist() + self.df['target'].tolist()), columns=['node'])
            num_features = len(self.df['features'].iloc[0])
            self.node_df['features'] = self.node_df.apply(lambda _: np.zeros(num_features), axis=1)
            self.node_df['label'] = 0

        self.edge_features = np.stack(self.df['features'].values)
        self.edge_features = np.vstack([np.zeros((1, self.edge_features.shape[1])), self.edge_features])
        max_node_idx = max(self.df['source'].max(), self.df['target'].max())
        self.node_features = np.zeros((max_node_idx + 1, self.edge_features.shape[1]))
        self.node_features[self.node_df['node'].values] = np.array(self.node_df['features'].tolist())
        self.df = self.df[['source', 'target', 'timestamp', 'label']]
        self.df.index += 1

        save_object((self.df, self.edge_features, self.node_features, self.column2map), dataset_path)

    @abstractmethod
    def get_data(self):
        pass

    def setup(self, stage=None):
        val_time, test_time = list(np.quantile(self.df['timestamp'], [0.70, 0.85]))

        nodes = np.unique(np.hstack([self.df['source'].values, self.df['target'].values]))
        test_df = self.df[val_time < self.df['timestamp']]
        test_nodes = np.unique(np.hstack([test_df['source'].values, test_df['target'].values]))
        test_nodes = np.random.choice(test_nodes, int(0.1 * len(nodes)), replace=False)

        train_mask = (~self.df['source'].isin(test_nodes) & ~self.df['target'].isin(test_nodes)) & (self.df['timestamp'] <= val_time)
        val_mask = (val_time < self.df['timestamp']) & (self.df['timestamp'] <= test_time)
        test_mask = test_time < self.df['timestamp']

        ### Initialize the data structure for graph and edge sampling
        # build the graph for fast query
        # graph only contains the training data (with 10% nodes removal)
        adj_list = [[] for _ in range(np.max(nodes) + 1)]
        for row in self.df[train_mask].itertuples():
            adj_list[row.source].append((row.target, row.Index, row.timestamp))
            adj_list[row.target].append((row.source, row.Index, row.timestamp))
        self.train_ngh_finder = NeighborFinder(adj_list, uniform=self.uniform)

        # full graph with all the data for the test and validation purpose
        full_adj_list = [[] for _ in range(np.max(nodes) + 1)]
        for row in self.df.itertuples():
            full_adj_list[row.source].append((row.target, row.Index, row.timestamp))
            full_adj_list[row.target].append((row.source, row.Index, row.timestamp))
        self.full_ngh_finder = NeighborFinder(full_adj_list, uniform=self.uniform)

        # define the new nodes sets for testing inductiveness of the model
        train_nodes = np.unique(np.hstack([self.df[train_mask]['source'].values, self.df[train_mask]['target'].values]))
        new_edge_mask = ~self.df['source'].isin(train_nodes) | ~self.df['target'].isin(train_nodes)

        self.train_dataset = GeneralDataset(self.df[train_mask])
        self.val_datasets = [GeneralDataset(self.df[val_mask & ~new_edge_mask]), GeneralDataset(self.df[val_mask & new_edge_mask])]
        self.test_datasets = [GeneralDataset(self.df[test_mask & ~new_edge_mask]), GeneralDataset(self.df[test_mask & new_edge_mask])]

        self.edge_list = self.df[train_mask][['source', 'target']].values
        self.train_edge_features = np.array(self.edge_features[1:][train_mask.tolist()], dtype=np.float)
        self.timestamps = self.df[train_mask]['timestamp'].values

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return [DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers) for val_dataset in self.val_datasets]

    def test_dataloader(self):
        return [DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers) for test_dataset in self.test_datasets]

    def quantile_and_factorize(self, df, quantile_columns=None, factorize_columns=None, merge=False):
        # quantile
        if quantile_columns is not None:
            for column in quantile_columns:
                data = df[column][~pd.isna(df[column])].astype('int').values
                df[column][~pd.isna(df[column])] = self._quantization(data, nbins=100)

        # factorize
        column2map = OrderedDict()
        if factorize_columns is not None:
            if merge:
                values = []
                for column in factorize_columns:
                    values += df[column].tolist()
                column_map = {x: i for i, x in enumerate(np.unique(values), 1)}
                for column in factorize_columns:
                    df[column] = df[column].map(column_map)
                    column2map[column] = column_map
            else:
                last_index = 1
                for column in factorize_columns:
                    column_map = {x: i for i, x in enumerate(df[column].unique(), last_index)}
                    df[column] = df[column].map(column_map)
                    last_index = last_index + len(column_map)
                    column2map[column] = column_map

        return df, column2map

    def _quantization(self, data, nbins):
        qtls = np.arange(0.0, 1.0 + 1 / nbins, 1 / nbins)
        bin_edges = np.unique(np.quantile(data, qtls, axis=0))
        quant_data = np.zeros(data.shape[0])
        for i, x in enumerate(data):
            quant_data[i] = np.digitize(x, bin_edges)
        quant_data = quant_data.clip(1, nbins) - 1
        return quant_data


class GeneralDataset(Dataset):
    def __init__(self, df):
        super(GeneralDataset, self).__init__()
        self.df = df
        self.negative_sampler = RandEdgeSampler(df['source'].values, df['target'].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index].name
        _, fake_target = self.negative_sampler.sample(1)

        return self.df.loc[row, 'source'], self.df.loc[row, 'target'], fake_target[0], self.df.loc[row, 'timestamp']

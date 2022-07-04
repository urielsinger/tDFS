import os
import pandas as pd
import numpy as np
import wget
import zipfile

from tDFS.datasets.Abstract import AbstractDataModule
from tDFS.constants import DATA_PATH


class BookingDataModule(AbstractDataModule):
    """
    DataModule handling data for the booking datasets.
    Please find the datasets in: https://www.bookingchallenge.com/
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def get_data(self):
        file_path = os.path.join(DATA_PATH, f'{self.dataset}.zip')
        if not os.path.exists(file_path):
            wget.download(f'https://035105f7-ae32-47b6-a25b-87af7924c7ea.filesusr.com/archives/3091be_233bace50f3f48fba40547a89443a96e.zip?dn=booking_dataset.zip', file_path)
            zipfile.ZipFile(file_path).extractall(os.path.join(DATA_PATH, self.dataset))
            os.remove(file_path)
        train_df = pd.read_csv(os.path.join(DATA_PATH, 'booking', 'train_set.csv'), parse_dates=['checkin', 'checkout'])
        train_df['dataset'] = 'train'
        test_df = pd.read_csv(os.path.join(DATA_PATH, 'booking', 'test_set.csv'), parse_dates=['checkin', 'checkout'])
        test_df['dataset'] = test_df['city_id'].map(lambda city_id: 'test' if city_id == 0 else 'train')
        labels_df = pd.read_csv(os.path.join(DATA_PATH, 'booking', 'ground_truth.csv'))
        test_df.loc[test_df['dataset'] == 'test', 'city_id'] = test_df.loc[test_df['dataset'] == 'test', 'utrip_id'].map({row['utrip_id']: row['city_id'] for _, row in labels_df.iterrows()})
        test_df.loc[test_df['dataset'] == 'test', 'hotel_country'] = test_df.loc[test_df['dataset'] == 'test', 'utrip_id'].map({row['utrip_id']: row['hotel_country'] for _, row in labels_df.iterrows()})
        df = pd.concat([train_df, test_df])
        df = df.sort_values('checkin')

        df['relative_position'] = df.groupby('utrip_id')['checkin'].rank(method='min', ascending=False)
        df['relative_trip_position'] = df.groupby('user_id')['utrip_id'].rank(method='dense', ascending=False)
        df['duration'] = (df['checkout'] - df['checkin']).map(lambda x: x.days)
        df['timestamp'] = df['checkin'].map(lambda x: x.timestamp())
        del df['checkin'], df['checkout'], df['dataset']
        features = ['relative_position', 'relative_trip_position', 'duration']
        for feature in ['device_class', 'booker_country']:
            df[feature] = df[feature].factorize()[0]
            feature_values = [f'{feature}_{i}' for i in range(df[feature].max() + 1)]
            feature_mat = np.zeros((len(df), len(feature_values)))
            feature_mat[np.arange(len(df)), df[feature].values] = 1
            df[feature_values] = feature_mat
            features += feature_values
        df['pad'] = 0
        features += ['pad']
        df['features'] = df.apply(lambda row: row[features].values, axis=1)
        df['label'] = 0
        df = df.rename({'user_id': 'source', 'city_id': 'target'}, axis=1)
        df = df[['source', 'target', 'timestamp', 'label', 'features']]
        df = df.reset_index()

        df, column2map = self.quantile_and_factorize(df, factorize_columns=['source', 'target'])

        return df, None, column2map

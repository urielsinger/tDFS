import os
import pandas as pd
import numpy as np
import wget
import zipfile

from tDFS.datasets.Abstract import AbstractDataModule
from tDFS.constants import DATA_PATH


class MovieLensDataModule(AbstractDataModule):
    """
    DataModule handling data for the MovieLens dataset.
    Please find the datasets in: https://grouplens.org/datasets/movielens/
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def get_data(self):
        file_path = os.path.join(DATA_PATH, 'ml-100k.zip')
        if not os.path.exists(file_path):
            wget.download(f'https://files.grouplens.org/datasets/movielens/ml-100k.zip', file_path)
            zipfile.ZipFile(file_path).extractall(DATA_PATH)
            os.remove(file_path)
        file_path = os.path.join(DATA_PATH, 'ml-1m.zip')
        if self.dataset.endswith('1m') and not os.path.exists(file_path):
            wget.download(f'https://files.grouplens.org/datasets/movielens/ml-1m.zip', file_path)
            zipfile.ZipFile(file_path).extractall(DATA_PATH)
            os.remove(file_path)

        genres = pd.read_csv(os.path.join(DATA_PATH, 'ml-100k', 'u.genre'), delimiter='|', header=None)[0].tolist()
        if self.dataset.endswith('100k'):
            edges_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'u.data'), delimiter='\t', header=None, names=['source', 'target', 'label', 'timestamp'])

            movie_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'u.item'), delimiter='|', header=None, names=['target', 'movie title', 'release date', 'video release date', 'IMDb URL'] + genres, encoding='latin-1')

            occupation = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'u.occupation'), delimiter='|', header=None)[0].values
            user_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'u.user'), delimiter='|', header=None, names=['source', 'age', 'gender', 'occupation', 'zip code'])
            user_df[occupation] = user_df.apply(lambda row: np.array(occupation == row['occupation'], dtype=int), result_type='expand', axis=1)
        elif self.dataset.endswith('1m'):
            edges_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'ratings.dat'), delimiter='::', header=None, names=['source', 'target', 'label', 'timestamp'])

            movie_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'movies.dat'), delimiter='::', header=None, names=['target', 'title', 'genres'], encoding='latin-1')
            movie_df[genres] = movie_df.apply(lambda row: np.array(np.isin(genres, row['genres'].split('|')), dtype=int), result_type='expand', axis=1)

            user_df = pd.read_csv(os.path.join(DATA_PATH, self.dataset, 'users.dat'), delimiter='::', header=None, names=['source', 'gender', 'age', 'occupation', 'zip code'])
            occupation = np.arange(user_df['occupation'].max() + 1)
            occupation_mat = np.zeros((len(user_df), len(occupation)))
            occupation_mat[np.arange(len(user_df)), user_df['occupation'].values] = 1
            user_df[occupation] = occupation_mat

        movie_df = movie_df[['target'] + genres]
        user_df['gender'] = user_df['gender'].factorize()[0]
        user_df = user_df[['source', 'age', 'gender'] + list(occupation)]

        df = edges_df.set_index('source').join(user_df.set_index('source')).reset_index()
        df = df.set_index('target').join(movie_df.set_index('target'), rsuffix='_y').reset_index()

        df, column2map = self.quantile_and_factorize(df, factorize_columns=['source', 'target'])
        df['features'] = df.apply(lambda row: np.array(row.values[4:]), axis=1)

        df = df[['source', 'target', 'timestamp', 'label', 'features']]

        return df, None, column2map

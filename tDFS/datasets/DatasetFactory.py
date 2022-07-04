import pytorch_lightning as pl

from tDFS.datasets import *

class DatasetFactory(pl.LightningDataModule):
    @classmethod
    def get_datamodule(cls, hparams):
        if hparams.dataset == 'wikipedia':
            return WikipediaDataModule(hparams.dataset,
                                       uniform=hparams.uniform,
                                       batch_size=hparams.batch_size,
                                       num_workers=hparams.num_workers)
        elif hparams.dataset == 'soc-redditHyperlinks-body':
            return RedditDataModule(hparams.dataset,
                                    uniform=hparams.uniform,
                                    batch_size=hparams.batch_size,
                                    num_workers=hparams.num_workers)
        elif hparams.dataset == 'act-mooc':
            return MOOCDataModule(hparams.dataset,
                                  uniform=hparams.uniform,
                                  batch_size=hparams.batch_size,
                                  num_workers=hparams.num_workers)
        elif hparams.dataset in ['ml-100k', 'ml-1m']:
            return MovieLensDataModule(hparams.dataset,
                                       uniform=hparams.uniform,
                                       batch_size=hparams.batch_size,
                                       num_workers=hparams.num_workers)
        elif hparams.dataset == 'booking':
            return BookingDataModule(hparams.dataset,
                                   uniform=hparams.uniform,
                                   batch_size=hparams.batch_size,
                                   num_workers=hparams.num_workers)

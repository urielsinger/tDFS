from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from tDFS.datasets.DatasetFactory import DatasetFactory
from tDFS.modules.tDFS import tDFSModule
from tDFS.config import parser


if __name__ == '__main__':
    hparams = parser.parse_args()
    pl.trainer.seed_everything(hparams.seed)

    logger = WandbLogger(save_dir=hparams.cache_path,
                         name='tDFS_' + hparams.dataset,
                         version=datetime.now().strftime('%y%m%d_%H%M%S.%f'),
                         project='tDFS',
                         config=hparams,
                         )

    datamodule = DatasetFactory.get_datamodule(hparams)
    datamodule.prepare_data()
    datamodule.setup()
    model = tDFSModule(datamodule.node_features, datamodule.edge_features, hparams)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.max_epochs,
                         logger=logger,
                         num_sanity_val_steps=0,
                         log_every_n_steps=1,
                         callbacks=[lr_monitor])
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

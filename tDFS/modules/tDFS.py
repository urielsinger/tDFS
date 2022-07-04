import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from tDFS.models import tDFS

class tDFSModule(pl.LightningModule):
    def __init__(self, node_features, edge_features, hparams):
        """
        pytorch_lightning module handling the training of tDFS
        """
        super(tDFSModule, self).__init__()

        self.learning_rate = hparams.learning_rate
        self.weight_decay = hparams.weight_decay
        self.max_neighbors = hparams.max_neighbors

        self.model = tDFS(node_features, edge_features,
                           attn_mode=hparams.attn_mode,
                           use_time=hparams.time,
                           bfs_method=hparams.bfs_method,
                           path_agg=hparams.path_agg,
                           paths_agg=hparams.paths_agg,
                           num_layers=hparams.num_hops,
                           n_head=hparams.n_heads,
                           dropout=hparams.dropout,
                           seq_len=hparams.max_neighbors,
                           alpha=hparams.alpha)

        self.criterion = nn.BCELoss()

    def forward(self, source, target, fake_target, timestamp):
        return self.model.contrast(source.long(), target.long(), fake_target.long(), timestamp.float())

    def step(self, batch, name):
        source, target, fake_target, timestamp = batch

        pos_logits, neg_logits = self(source, target, fake_target, timestamp)

        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]).to(self.device)
        logits = torch.cat([pos_logits, neg_logits])
        preds = logits.round()

        loss = self.criterion(logits, labels)
        self.log(f'{name}/loss', loss)

        logits = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()
        self.log(f'{name}/Accuracy', (labels == preds).mean())
        self.log(f'{name}/Average Precision', average_precision_score(labels, logits))
        self.log(f'{name}/F1 Score', f1_score(labels, preds))
        self.log(f'{name}/AUC', roc_auc_score(labels, logits))

        return loss

    def on_train_epoch_start(self) -> None:
        self.model.ngh_finder = self.trainer.datamodule.train_ngh_finder
    def on_validation_epoch_start(self) -> None:
        self.model.ngh_finder = self.trainer.datamodule.full_ngh_finder
    def on_test_epoch_start(self) -> None:
        self.model.ngh_finder = self.trainer.datamodule.full_ngh_finder

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        return self.step(batch, name='train')

    def validation_step(self, batch: dict, batch_idx: int, dataset_idx: int) -> dict:
        return self.step(batch, name=f'val')

    def test_step(self, batch: dict, batch_idx: int, dataset_idx: int):
        return self.step(batch, name=f'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

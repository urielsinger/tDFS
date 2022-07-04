import sys
import argparse
from tDFS.constants import CACHE_PATH

_using_debugger = getattr(sys, "gettrace", None)() is not None

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='wikipedia', choices=['wikipedia', 'soc-redditHyperlinks-body', 'act-mooc', 'ml-100k', 'ml-1m', 'booking'], help='name of the datasets')
parser.add_argument('--cache_path', type=str, default=CACHE_PATH, help='cache path')

parser.add_argument('--bfs_method', type=str, default='attn', choices=['mean', 'attn', 'lstm'], help='bfs aggregation method')
parser.add_argument('--attn_mode', type=str, default='prod', choices=['prod', 'map'], help='use dot product attention or mapping based')
parser.add_argument('--path_agg', type=str, default='attn', choices=['mean', 'attn', 'lstm'], help='path aggregation method')
parser.add_argument('--paths_agg', type=str, default='attn', choices=['attn', 'mean'], help='paths aggregation method')

parser.add_argument('--num_hops', type=int, default=2, help='number of temporal neighbor hops')
parser.add_argument('--max_neighbors', type=int, default=20, help='maximum number of neighbors')
parser.add_argument('--uniform', action='store_true', default=True, help='take uniform sampling from temporal neighbors')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads in an attention layer')
parser.add_argument('--time', type=str, default='time', choices=['time', 'pos', 'empty'], help='how to use time information')
parser.add_argument('--alpha', type=float, default=0.5, help='balance between bfs and dfs')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--max_epochs', type=int, default=10, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=200, help='number of samples in batch')

parser.add_argument('--gpus', type=int, default=1, help='gpus parameter used for pytorch_lightning')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
if _using_debugger:
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
else:
    parser.add_argument('--num_workers', type=int, default=6, help='number of workers for dataloader')
parser.add_argument('--debug', action='store_true', default=_using_debugger, help='run in debug mode')

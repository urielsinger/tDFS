# tDFS: Temporal Graph Neural Network Leveraging DFS

### Uriel Singer, Haggai Roitman, Ido Guy and Kira Radinsky. 

> **Abstract:**
> Temporal graph neural networks (temporal GNNs) have been widely researched, reaching state-of-the-art results on multiple prediction tasks. A common approach employed by most previous works is to apply a layer that aggregates information from the historical neighbors of a node. Taking a different research direction, in this work, we propose tDFS -- a novel temporal GNN architecture. tDFS applies a layer that efficiently aggregates information from temporal paths to a given (target) node in the graph. For each given node, the aggregation is applied in two stages: (1) A single representation is learned for each temporal path ending in that node, and (2) all path representations are aggregated into a final node representation. Overall, our goal is not to add new information to a node, but rather observe the same exact information in a new perspective. This  allows our model to directly observe patterns that are path-oriented rather than neighborhood-oriented. This can be thought as a Depth-First Search (DFS) traversal over the temporal graph, compared to the popular Breath-First Search (BFS) traversal that is applied in previous works. We evaluate tDFS over multiple link prediction tasks and show its favorable performance compared to state-of-the-art baselines. To the best of our knowledge, we are the first to apply a temporal-DFS neural network.

This repository provides a reference implementation of tDFS as described in the paper.

## Requirements
 - python==3.9.5
 - torch==1.10.0
 - pytorch-lightning==1.5.2
 - torch-geometric==2.0.2
 - wandb==0.12.7
 - pandas==1.3.4
 - sklearn==1.0.1
 - wget==3.2

## Data
During the first run, the dataset is automatically downloaded and cached.

## Usage
```bash
python run.py \
    --dataset=${dataset} \
    --bfs_method=${bfs_method} \
    --attn_mode=${attn_mode} \
    --path_agg=${path_agg} \
    --paths_agg=${paths_agg} \
    --num_hops=${num_hops} \
    --max_neighbors=${max_neighbors} \
    --uniform
    --n_heads=${n_heads} \
    --time=${time} \
    --alpha=${alpha} \
    --gpus=${gpus} \
    --num_workers=${num_workers} \
```

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{tdfs,
  title     = {tDFS: Temporal Graph Neural Network Leveraging DFS},
  author    = {Singer, Uriel and Roitman, Haggai and Guy, Ido and Radinsky, Kira},
}
```

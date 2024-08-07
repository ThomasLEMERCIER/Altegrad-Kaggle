import torch
import random
import numpy as np
from dataclasses import dataclass
from torch_geometric.data import Data
from torch_geometric import utils as tg_utils


@dataclass(frozen=True)
class GraphDataAugParams:
    lambda_aug: float
    min_aug: int
    max_aug: int

    p_edge_pertubation: float
    edge_pertubation: float
    uniform_edge_number: bool

    p_graph_sampling: float
    graph_sampling: float

    p_features_noise: float
    features_noise: float

    p_features_shuffling: float
    features_shuffling: float

    p_features_masking: float
    features_masking: float

    p_khop_subgraph: float


@dataclass(frozen=True)
class DataAugParams:
    graph_params: GraphDataAugParams


def edge_pertubation(x, edge_index, p, uniform_edge_number=False):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p: probability of edge pertubation

    returns: perturbed edge index i.e. for an adjacency matrix A, A' = A XOR E, E bernoulli random variable matrix
    """
    if uniform_edge_number:
        p = random.random() * p

    edge_index, _ = tg_utils.dropout_edge(edge_index, p=p, force_undirected=True)
    # p is the proportion of new edges to add
    if int(edge_index.shape[1] * p) <= 1:  # if p is too small to add any edges
        return x, edge_index

    edge_index, _ = tg_utils.add_random_edge(
        edge_index, p=p, force_undirected=True, num_nodes=x.shape[0]
    )

    return x, edge_index


def graph_sampling(x, edge_index, p):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p: probability of removing a node

    returns: sampled node from the graph
    """
    num_nodes_to_keep = torch.randint(
        low=max(int(x.shape[0] * p), 1), high=x.shape[0] + 1, size=(1,)
    ).item()
    nodes_to_keep = torch.randperm(x.shape[0])[:num_nodes_to_keep]
    edge_index, _ = tg_utils.subgraph(
        subset=nodes_to_keep,
        edge_index=edge_index,
        num_nodes=x.shape[0],
        relabel_nodes=True,
    )

    return x[nodes_to_keep], edge_index


def k_hop_subgraph(x, edge_index):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)

    returns: k-hop subgraph
    """
    start_node = random.randint(0, x.shape[0] - 1)
    k = random.randint(5, 10)

    sub_nodes, sub_edge_index, _, _ = tg_utils.k_hop_subgraph(
        start_node, k, edge_index, relabel_nodes=True, num_nodes=x.shape[0]
    )

    return x[sub_nodes], sub_edge_index


def features_corruption(x, edge_index, std):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    std: standard deviation of gaussian noise

    returns: corrupted node features
    """
    return x + torch.randn(x.shape) * std, edge_index


def features_shuffling(x, edge_index, p_features):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p_features: proportion of features to shuffle

    returns: shuffled node features
    """
    n = x.shape[1]
    m = int(n * p_features)
    permuted_indices = torch.randperm(n)[: 2 * m].view(m, 2)

    perm = torch.arange(n)
    perm[permuted_indices[:, 0]] = permuted_indices[:, 1]
    perm[permuted_indices[:, 1]] = permuted_indices[:, 0]

    return x[:, perm], edge_index


def features_masking(x, edge_index, p):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p: probability of masking a feature

    returns: masked node features
    """
    mask = torch.rand(x.shape) > p
    return x * mask, edge_index


AUGS = [
    edge_pertubation,
    graph_sampling,
    features_corruption,
    features_shuffling,
    features_masking,
    k_hop_subgraph,
]
NB_AUGMENTATIONS = len(AUGS)


def random_graph_data_aug(x, edge_index, params: GraphDataAugParams):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    params: data augmentation parameters
    """
    if params.lambda_aug == 0:
        return x, edge_index

    n_aug = np.clip(
        np.random.poisson(params.lambda_aug),
        params.min_aug,
        min(params.max_aug, NB_AUGMENTATIONS),
    )
    which_aug = np.random.choice(
        NB_AUGMENTATIONS,
        n_aug,
        replace=False,
        p=[
            params.p_edge_pertubation,
            params.p_graph_sampling,
            params.p_features_noise,
            params.p_features_shuffling,
            params.p_features_masking,
            params.p_khop_subgraph,
        ],
    )

    aug_param_list = [
        (params.edge_pertubation, params.uniform_edge_number),
        (params.graph_sampling,),
        (params.features_noise,),
        (params.features_shuffling,),
        (params.features_masking,),
        tuple(),
    ]

    for aug in which_aug:
        x, edge_index = AUGS[aug](x, edge_index, *aug_param_list[aug])

    return x, edge_index


def random_data_aug(data, params: DataAugParams):
    x, edge_index = random_graph_data_aug(data.x, data.edge_index, params.graph_params)
    return Data(
        x=x,
        edge_index=edge_index,
        input_ids=data.input_ids,
        attention_mask=data.attention_mask,
    )

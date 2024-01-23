import torch
import random
from dataclasses import dataclass
from torch_geometric import utils as tg_utils
from torch_geometric.data import Data

@dataclass(frozen=True)
class GraphDataAugParams:
    p_edge_pertubation: float
    p_graph_sampling: float
    features_noise: float
    p_features_shuffling: float
    p_features_masking: float

@dataclass(frozen=True)
class DataAugParams:
    graph_params: GraphDataAugParams

def edge_pertubation(x, edge_index, p):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p: probability of edge pertubation
    
    returns: perturbed edge index i.e. for an adjacency matrix A, A' = A XOR E, E bernoulli random variable matrix
    """
    # check if the graph has enough nodes
    if x.shape[0] < 2:
        return x, edge_index
    edge_index, _ = tg_utils.dropout_edge(edge_index, p=p, force_undirected=True)
    edge_index, _ = tg_utils.add_random_edge(edge_index, p=p, force_undirected=True, num_nodes=x.shape[0])
    return x, edge_index

def graph_sampling(x, edge_index, p):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p: probability of removing a node
    
    returns: sampled node from the graph
    """
    to_keep = torch.randperm(x.shape[0])[:-int(x.shape[0] * p)]
    edge_index, _ = tg_utils.subgraph(subset=to_keep, edge_index=edge_index, num_nodes=x.shape[0])
    return x[to_keep], edge_index

def features_corruption(x, edge_index, std):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    std: standard deviation of gaussian noise
    
    returns: corrupted node features
    """
    return x + torch.randn(x.shape) * std, edge_index

def features_shuffling(x, edge_index):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    
    returns: shuffled node features
    """
    return x[torch.randperm(x.shape[0])], edge_index

def features_masking(x, edge_index, p):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    p: probability of masking a feature
    
    returns: masked node features
    """
    mask = torch.rand(x.shape) < p
    return x * mask, edge_index

def random_graph_data_aug(x, edge_index, params: GraphDataAugParams):
    """
    x: node features torch tensor of shape (num_nodes, num_node_features)
    edge_index: edge index torch tensor of shape (2, num_edges)
    p_edge_pertubation: probability of edge pertubation
    p_graph_sampling: probability of removing a node
    std: standard deviation of gaussian noise
    p_features_shuffling: probability of shuffling features
    p_features_masking: probability of masking a feature
    """

    # Edge pertubation
    if params.p_edge_pertubation > 0:
        x, edge_index = edge_pertubation(x, edge_index, params.p_edge_pertubation)
    
    # Graph sampling
    if params.p_graph_sampling > 0:
        x, edge_index = graph_sampling(x, edge_index, params.p_graph_sampling)
    
    # Features corruption
    if params.features_noise > 0:
        x, edge_index = features_corruption(x, edge_index, params.features_noise)
    
    # Features shuffling
    if params.p_features_shuffling > 0:
        if random.random() < params.p_features_shuffling:
            x, edge_index = features_shuffling(x, edge_index)
    
    # Features masking
    if params.p_features_masking > 0:
        x, edge_index = features_masking(x, edge_index, params.p_features_masking)
    
    return x, edge_index

def random_data_aug(data, params: DataAugParams):
    x, edge_index = random_graph_data_aug(data.x, data.edge_index, params.graph_params)
    return Data(x=x, edge_index=edge_index, input_ids=data.input_ids, attention_mask=data.attention_mask)

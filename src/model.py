import torch
from torch import nn
from transformers import AutoModel, AutoModelForMaskedLM
from torch_geometric.nn.models import GAT, GCN
from torch_geometric.nn import global_mean_pool, SoftmaxAggregation
    
class GraphAttentionNetwork(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, num_layers, dropout):
        super(GraphAttentionNetwork, self).__init__()
        self.gat = GAT(in_channels=num_node_features, hidden_channels=graph_hidden_channels, out_channels=graph_hidden_channels, num_layers=num_layers, dropout=dropout, v2=True, norm="GraphNorm")

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)

class GraphConvolutionalNetwork(nn.Module):
    def __init__(self, num_node_features, graph_hidden_channels, num_layers, dropout):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gcn = GCN(in_channels=num_node_features, hidden_channels=graph_hidden_channels, out_channels=graph_hidden_channels, num_layers=num_layers, dropout=dropout, norm="GraphNorm")
    def forward(self, x, edge_index):
        return self.gcn(x, edge_index)
    
GNN_MODELS = {
    "gat": GraphAttentionNetwork,
    "gcn": GraphConvolutionalNetwork
}

class GraphEncoder(nn.Module):
    def __init__(self, model_name, num_node_features, graph_hidden_channels, nhid, nout, dropout=0, num_layers=3):
        super(GraphEncoder, self).__init__()
        self.gnn = GNN_MODELS[model_name](num_node_features, graph_hidden_channels, num_layers, dropout)

        self.head = nn.Sequential(
            nn.Linear(graph_hidden_channels, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nout),
        )

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.gnn(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.head(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, checkpoint=None, avg_pool=False):
        super(TextEncoder, self).__init__()
        self.avg_pool = avg_pool
        if checkpoint is None:
            self.main = AutoModel.from_pretrained(model_name)
        else:
            self.main = AutoModelForMaskedLM.from_pretrained(checkpoint).base_model

    def forward(self, input_ids, attention_mask):
        encoded_text = self.main(input_ids, attention_mask=attention_mask)
        if self.avg_pool:
            return torch.mean(encoded_text.last_hidden_state, dim=1)
        else:
            return encoded_text.last_hidden_state[:, 0, :]

class Model(nn.Module):
    def __init__(
        self,
        nlp_model_name,
        gnn_model_name,
        num_node_features,
        graph_hidden_channels,
        nhid,
        nout,
        gnn_dropout=0,
        gnn_num_layers=3,
        gnn_checkpoint=None,
        nlp_checkpoint=None,
        avg_pool_nlp=False,
    ):
        super(Model, self).__init__()

        self.graph_encoder = GraphEncoder(
            model_name=gnn_model_name,
            num_node_features=num_node_features,
            nout=nout,
            nhid=nhid,
            graph_hidden_channels=graph_hidden_channels,
            dropout=gnn_dropout,
            num_layers=gnn_num_layers
        )

        if gnn_checkpoint is not None:
            self.graph_encoder.load_state_dict(torch.load(gnn_checkpoint)["model_state_dict"])
        
        self.text_encoder = TextEncoder(model_name=nlp_model_name, checkpoint=nlp_checkpoint, avg_pool=avg_pool_nlp)

    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded

    def get_text_encoder(self):
        return self.text_encoder

    def get_graph_encoder(self):
        return self.graph_encoder

# Standard library imports
import yaml
import os.path as osp

# Related third party imports
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Local application/library specific imports
from .model import Model
from .dataset import GraphTextDataset
from .constants import CHECKPOINT_FOLDER, NODE_FEATURES_SIZE, ROOT_DATA, GT_PATH
from .data_aug import DataAugParams, GraphDataAugParams, random_data_aug

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(config):
    # === NLP model === #
    nlp_model_name = config["nlp_model_name"]
    custom_tokenizer = config.get("custom_tokenizer", False)
    nlp_pretrained = config.get("nlp_pretrained", False)

    # ==== GNN parameters ==== #
    gnn_model_name = config["gnn_model_name"]
    gnn_num_layers = config["gnn_num_layers"]
    gnn_dropout = config["gnn_dropout"]
    gnn_hdim = config["gnn_hdim"]
    mlp_hdim = config["mlp_hdim"]

    # ==== Output parameters ==== #
    nout = config["nout"]

    # ==== Loss/Model options ==== #
    avg_pool_nlp = config.get("avg_pool_nlp", False)

    # ==== NLP checkpoint ==== #
    nlp_checkpoint = config.get("nlp_checkpoint", None)
    if not nlp_pretrained:
        nlp_checkpoint = None
    else:
        nlp_checkpoint = osp.join(CHECKPOINT_FOLDER, "pretraining", nlp_model_name, nlp_checkpoint)

    model = Model(
        nlp_model_name=nlp_model_name,
        gnn_model_name=gnn_model_name,
        num_node_features=NODE_FEATURES_SIZE,
        graph_hidden_channels=gnn_hdim,
        nhid=mlp_hdim,
        nout=nout,
        gnn_dropout=gnn_dropout,
        gnn_num_layers=gnn_num_layers,
        nlp_checkpoint=nlp_checkpoint,
        avg_pool_nlp=avg_pool_nlp
    )

    return model

def load_optimizer(model, config):
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def load_checkpoint(model, optimizer, checkpoint_path):
    model_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])
    start_epoch = model_checkpoint["epoch"] + 1
    return model, optimizer, start_epoch

def load_model_from_checkpoint(model, checkpoint_path):
    model_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    return model

def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        checkpoint_path,
    )

def get_dataloaders(config, tokenizer, transform=None, transform_params=None, only_val=False):
    gt = np.load(GT_PATH, allow_pickle=True)[()]

    if only_val:
        val_dataset = GraphTextDataset(
            root=ROOT_DATA,
            gt=gt,
            split="val",
            tokenizer=tokenizer,
            nlp_model=config["nlp_model_name"],
            in_memory=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=1, drop_last=False)
        return val_loader
    
    train_dataset = GraphTextDataset(
        root=ROOT_DATA,
        gt=gt,
        split="train",
        tokenizer=tokenizer,
        nlp_model=config["nlp_model_name"],
        in_memory=True,
        transform=transform,
        transform_params=transform_params
    )
    val_dataset = GraphTextDataset(
        root=ROOT_DATA,
        gt=gt,
        split="val",
        tokenizer=tokenizer,
        nlp_model=config["nlp_model_name"],
        in_memory=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=1)

    return train_loader, val_loader

def get_transform(config):
    g_paramrs = GraphDataAugParams(
        p_edge_pertubation=config["edge_pertubation"],
        p_graph_sampling=config["graph_sampling"],
        features_noise=config["features_noise"],
        p_features_shuffling=config["features_shuffling"],
        p_features_masking=config["features_masking"],
    )

    return random_data_aug, DataAugParams(graph_params=g_paramrs)

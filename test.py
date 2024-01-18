# Standard library imports
import os
import time
import logging
import argparse
import os.path as osp

# Related third-party imports
import torch
import numpy as np
from transformers import AutoTokenizer
from torch_geometric.loader import DataLoader
from yaml import safe_load

# Local application/library specific imports
from src.constants import *
from src.dataset import GraphTextDataset
from src.model import Model
from src.training import validation_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline.yaml", help="Name of config file")
    parser.add_argument("--weights", required=True, help="Path to weights")

    args = parser.parse_args()
    config_path = osp.join("configs", args.config)
    config = safe_load(open(config_path, "r"))

    run_name = config["name"]

    model_name = config["model_name"]
    batch_size = config["batch_size"]
    nb_epochs = config["nb_epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    gnn_hdim = config["gnn_hdim"]
    mlp_hdim = config["mlp_hdim"]

    nout = config["nout"]

    checkpoint_path = osp.join("checkpoints", run_name)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    logging.basicConfig(
        filename=osp.join(checkpoint_path, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    root = ROOT_DATA
    gt = np.load(GT_PATH, allow_pickle=True)[()]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets")
    loading_time = time.time()
    train_dataset = GraphTextDataset(
        root=root,
        gt=gt,
        split="train",
        tokenizer=tokenizer,
        nlp_model=model_name,
        in_memory=True,
    )
    val_dataset = GraphTextDataset(
        root=root,
        gt=gt,
        split="val",
        tokenizer=tokenizer,
        nlp_model=model_name,
        in_memory=True,
    )
    print("Loading time: ", time.time() - loading_time)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        model_name=model_name,
        num_node_features=NODE_FEATURES_SIZE,
        nout=nout,
        nhid=mlp_hdim,
        graph_hidden_channels=gnn_hdim,
    )

    model.load_state_dict(torch.load(args.weights))
    model.to(device)

    validation_loss = validation_epoch(val_loader, device, model)

    logging.info(f"Validation loss: {validation_loss}")

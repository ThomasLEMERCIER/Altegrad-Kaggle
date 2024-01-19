# Standard library imports
import os
import time
import logging
import argparse
import os.path as osp

# Related third-party imports
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd

from yaml import safe_load

# Local application/library specific imports
from src.constants import *
from src.dataset import GraphTextDataset, TextDataset, GraphDataset
from src.model import Model
from src.training import validation_epoch
from src.evaluation import text_inference, graph_inference


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
        filename=osp.join(checkpoint_path, "test.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    root = ROOT_DATA
    gt = np.load(GT_PATH, allow_pickle=True)[()]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets")
    loading_time = time.time()

    val_dataset = GraphTextDataset(
        root=root,
        gt=gt,
        split="val",
        tokenizer=tokenizer,
        nlp_model=model_name,
        in_memory=False,
    )
    print("Loading time: ", time.time() - loading_time)

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

    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    validation_loss, validation_lrap = validation_epoch(val_loader, device, model)

    print(f"Validation loss: {validation_loss}")
    print(f"Validation LRAP: {validation_lrap}")

    # ----  test inference  ----

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    text_dataset = TextDataset(
        root=root, test_file="test_text", tokenizer=tokenizer, nlp_model=model_name
    )
    text_dataloader = TorchDataLoader(
        text_dataset, batch_size=batch_size, shuffle=False
    )

    text_embeddings = text_inference(text_dataloader, device, text_model)

    graph_dataset = GraphDataset(root=root, gt=gt, split="test_cids")
    graph_dataloader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = graph_inference(graph_dataloader, device, graph_model)

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv("submission.csv", index=False)

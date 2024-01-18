# Standard library imports
import os
import time
import logging
import argparse
import os.path as osp

# Related third-party imports
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.loader import DataLoader
from yaml import safe_load

# Local application/library specific imports
from src.constants import *
from src.dataset import GraphTextDataset
from src.loss import contrastive_loss
from src.model import Model

def train_epoch(train_loader, device, model, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        batch.pop("input_ids")
        batch.pop("attention_mask")
        graph_batch = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        graph_batch = graph_batch.to(device)

        optimizer.zero_grad()

        graph_embeddings, text_embeddings = model(graph_batch, input_ids, attention_mask)
        loss = contrastive_loss(graph_embeddings, text_embeddings)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    return average_loss

def validation_epoch(val_loader, device, model):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            batch.pop("input_ids")
            batch.pop("attention_mask")
            graph_batch = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_batch = graph_batch.to(device)


            graph_embeddings, text_embeddings = model(graph_batch, input_ids, attention_mask)
            loss = contrastive_loss(graph_embeddings, text_embeddings)
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline.yaml", help="Name of config file")

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

    logging.info(f"Run name: {run_name}")

    root = ROOT_DATA
    gt = np.load(GT_PATH, allow_pickle=True)[()]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading datasets")
    loading_time = time.time()
    train_dataset = GraphTextDataset(root=root, gt=gt, split="train", tokenizer=tokenizer, nlp_model=model_name, in_memory=False)
    val_dataset = GraphTextDataset(root=root, gt=gt, split="val", tokenizer=tokenizer, nlp_model=model_name, in_memory=False)
    print("Loading time: ", time.time() - loading_time)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        model_name=model_name,
        num_node_features=NODE_FEATURES_SIZE,
        nout=nout,
        nhid=mlp_hdim,
        graph_hidden_channels=gnn_hdim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    best_validation = np.inf

    for e in range(1, nb_epochs + 1):
        print("----- EPOCH {} -----".format(e))
        trainning_loss = train_epoch(train_loader, device, model, optimizer)
        validation_loss = validation_epoch(val_loader, device, model)

        logging.info(
            f"Epoch {e}: Training loss: {trainning_loss}, Validation loss: {validation_loss}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation": best_validation,
                "epoch": e,
                "validation_loss": validation_loss,
                "training_loss": trainning_loss,
            }
            save_path = osp.join(checkpoint_path, f"checkpoint_{e}.pt")
            torch.save(checkpoint, save_path)
    
    logging.info(f"Best validation loss: {best_validation}")

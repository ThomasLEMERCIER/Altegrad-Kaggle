# Standard library imports
import argparse
import datetime
import logging
import os
import os.path as osp
import time

import numpy as np

# Related third-party imports
import torch
import wandb
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer
from yaml import safe_load

# Local application/library specific imports
from src.constants import *
from src.dataset import GraphTextDataset
from src.model import Model
from src.training import train_epoch, validation_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline.yaml", help="Name of config file")
    parser.add_argument("--wandb", action="store_true", help="Use wandb")

    args = parser.parse_args()
    config_path = osp.join("configs", args.config)
    config = safe_load(open(config_path, "r"))

    # ==== Run name ==== #
    run_name = (
        config["name"]
        + "_("
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ")"
    )

    # ==== Nlp model parameters ==== #
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

    # ==== Training parameters ==== #
    batch_size = config["batch_size"]
    nb_epochs = config["nb_epochs"]
    lr = config["lr"]
    weight_decay = config["weight_decay"]

    # ==== Loss/Model options ==== #
    norm_loss = config.get("norm_loss", False)
    avg_pool_nlp = config.get("avg_pool_nlp", False)

    # ==== NLP checkpoint ==== #
    nlp_checkpoint = config.get("nlp_checkpoint", None)
    if not nlp_pretrained:
        nlp_checkpoint = None
    else:
        nlp_checkpoint = osp.join(CHECKPOINT_FOLDER, "pretraining", nlp_model_name, nlp_checkpoint)

    # ==== Fine tuning ==== #
    fine_tune = config.get("fine_tuning", False)
    if fine_tune:
        run_name += "_finetune"
        checkpoint_name = config["checkpoint_name"]

    # ==== Checkpoint ==== #
    checkpoint_path = osp.join("checkpoints", run_name)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # ==== Logging ==== #
    logging.basicConfig(
        filename=osp.join(checkpoint_path, "train.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logging.info(f"Run name: {run_name}")

    if args.wandb:
        wandb.init(
            entity="thomas_l",
            project="Deep Node",
            name=run_name,
        )
        wandb.config.update(config)

    # ==== Data ==== #
    root = ROOT_DATA
    gt = np.load(GT_PATH, allow_pickle=True)[()]
    if custom_tokenizer:
        tokenizer_path = osp.join(
            CHECKPOINT_FOLDER, "pretraining", nlp_model_name, "tokenizer"
        )
    else:
        tokenizer_path = nlp_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    print("Loading datasets")
    loading_time = time.time()
    train_dataset = GraphTextDataset(
        root=root,
        gt=gt,
        split="train",
        tokenizer=tokenizer,
        nlp_model=nlp_model_name,
        in_memory=True,
    )
    val_dataset = GraphTextDataset(
        root=root,
        gt=gt,
        split="val",
        tokenizer=tokenizer,
        nlp_model=nlp_model_name,
        in_memory=True,
    )
    print("Loading time: ", time.time() - loading_time)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    # ==== Device ==== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Model ==== #
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
        avg_pool_nlp=avg_pool_nlp,
    ).to(device)

    # ==== Optimizer ==== #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )

    best_validation = np.inf
    start_epoch = 1

    if fine_tune:
        nlp_checkpoint = torch.load(osp.join("checkpoints", checkpoint_name))
        model.load_state_dict(nlp_checkpoint["model_state_dict"])
        optimizer.load_state_dict(nlp_checkpoint["optimizer_state_dict"])
        best_validation = nlp_checkpoint["best_validation"]
        start_epoch = nlp_checkpoint["epoch"] + 1
        print("Loaded checkpoint: ", checkpoint_name)
        print("Best validation loss: ", best_validation)
        print("Starting from epoch: ", start_epoch)

    for e in range(start_epoch, start_epoch + nb_epochs):
        print("----- EPOCH {} -----".format(e))
        trainning_loss = train_epoch(
            train_loader, device, model, optimizer, args.wandb, norm_loss
        )
        validation_loss, validation_lrap = validation_epoch(
            val_loader, device, model, norm_loss
        )

        if args.wandb:
            wandb.log(
                {
                    "training_loss": trainning_loss,
                    "validation_loss": validation_loss,
                    "validation_lrap": validation_lrap,
                }
            )

        logging.info(
            f"Epoch {e}: Training loss: {trainning_loss}, Validation loss, LRAP: {validation_loss}, {validation_lrap}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            nlp_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation": best_validation,
                "epoch": e,
                "validation_loss": validation_loss,
                "training_loss": trainning_loss,
            }
            save_path = osp.join(checkpoint_path, f"checkpoint_{e}.pt")
            torch.save(nlp_checkpoint, save_path)

    logging.info(f"Best validation loss: {best_validation}")

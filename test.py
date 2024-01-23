# Standard library imports
import os
import time
import logging
import argparse
import datetime
import os.path as osp

# Related third-party imports
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader as TorchDataLoader

# Local application/library specific imports
from src.constants import *
from src.training import validation_epoch
from src.dataset import GraphDataset, TextDataset
from src.evaluation import graph_inference, text_inference
from src.utils import load_config, load_model, get_dataloaders, load_model_from_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline.yaml", help="Name of config file")
    parser.add_argument("--weights", required=True, help="Path to weights")

    args = parser.parse_args()
    config_path = osp.join("configs", args.config)
    config = load_config(config_path)

    run_name = (
        config["name"]
        + "_("
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ")"
        + "_finetuning" if config["fine_tuning"] else ""
    )

    checkpoint_path = osp.join("checkpoints", run_name)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # ==== Logging ==== #
    logging.basicConfig(
        filename=osp.join(checkpoint_path, "test.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    # ==== Tokenizer ==== #
    if config["custom_tokenizer"]:
        tokenizer_path = osp.join(
            CHECKPOINT_FOLDER, "pretraining", config["nlp_model_name"], "tokenizer"
        )
    else:
        tokenizer_path = config["nlp_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ===== Dataloaders ===== #
    print("Loading datasets")
    loading_time = time.time()
    val_loader = get_dataloaders(
        config=config,
        tokenizer=tokenizer,
        only_val=True,
    )
    print("Loading time: ", time.time() - loading_time)
    
    # ==== Device ==== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Model ==== #
    model = load_model(config).to(device)
    model = load_model_from_checkpoint(model, args.weights)
    model.to(device)
    model.eval()

    validation_loss, validation_lrap = validation_epoch(val_loader, device, model)

    print(f"Validation loss: {validation_loss}")
    print(f"Validation LRAP: {validation_lrap}")

    # ====  test inference  ==== #
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    text_dataset = TextDataset(root=ROOT_DATA, test_file="test_text", tokenizer=tokenizer, nlp_model=config["nlp_model_name"])
    text_dataloader = TorchDataLoader(text_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    gt = np.load(GT_PATH, allow_pickle=True)[()]
    graph_dataset = GraphDataset(root=ROOT_DATA, gt=gt, split="test_cids")
    graph_dataloader = DataLoader(graph_dataset, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    text_embeddings = text_inference(text_dataloader, device, text_model)
    graph_embeddings = graph_inference(graph_dataloader, device, graph_model)

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv("submission.csv", index=False)

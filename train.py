# Standard library imports
import os
import time
import logging
import argparse
import datetime
import os.path as osp

# Related third-party imports
import torch
import wandb
from transformers import AutoTokenizer

# Local application/library specific imports
from src.constants import *
from src.training import train_epoch, validation_epoch
from src.utils import load_checkpoint, load_config, load_model, load_optimizer, get_dataloaders, save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="baseline.yaml", help="Name of config file")
    parser.add_argument("--wandb", action="store_true", help="Use wandb")

    args = parser.parse_args()
    config_path = osp.join("configs", args.config)
    config = load_config(config_path)

    # ==== Run name ==== #
    run_name = f"{config['name']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{('_finetuning' if config['fine_tuning'] else '')}"
    print("Run name: ", run_name)

    # ==== Checkpoint ==== #
    checkpoint_path = osp.join(CHECKPOINT_FOLDER, run_name)
    if not osp.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print("Checkpoint path: ", checkpoint_path)

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
    train_loader, val_loader = get_dataloaders(
        config=config,
        tokenizer=tokenizer,
        only_val=False,
    )
    print("Loading time: ", time.time() - loading_time)
    
    # ==== Device ==== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Model ==== #
    model = load_model(config).to(device)

    # ==== Optimizer ==== #
    optimizer = load_optimizer(model, config)
    best_validation_larp = 0
    start_epoch = 1
    nb_epochs = config["nb_epochs"]
    norm_loss = config["norm_loss"]

    if  config["fine_tuning"]:
        checkpoint_path = osp.join(CHECKPOINT_FOLDER, config["checkpoint_name"])
        model, optimizer, start_epoch = load_checkpoint(
            model, optimizer, checkpoint_path
        )

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
            f"Epoch {e}: Training loss: {trainning_loss}, Validation loss: {validation_loss}, LRAP: {validation_lrap}"
        )

        if validation_lrap > best_validation_larp:
            best_validation_larp = validation_lrap
            save_path = osp.join(checkpoint_path, f"checkpoint_{e}.pt")
            save_checkpoint(model, optimizer, e, save_path)

    logging.info(f"Best validation LARP: {best_validation_larp}")

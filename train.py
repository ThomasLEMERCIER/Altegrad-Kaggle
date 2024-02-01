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
from src.utils import (
    load_checkpoint,
    load_config,
    load_model,
    load_optimizer,
    get_dataloaders,
    save_checkpoint,
    get_scheduler,
    update_decay_scheduler,
    get_transform,
    get_top_k_scheduler,
    update_top_k,
)

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

    # ==== Transform ==== #
    transform, transform_params = get_transform(config)
    print("Transform params: ", transform_params)

    # ===== Dataloaders ===== #
    print("Loading datasets")
    loading_time = time.time()
    train_loader, val_loader = get_dataloaders(
        config=config,
        tokenizer=tokenizer,
        only_val=False,
        transform=transform,
        transform_params=transform_params,
    )
    print("Loading time: ", time.time() - loading_time)

    # ==== Device ==== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Model ==== #
    model = load_model(config).to(device)

    # ==== Optimizer ==== #
    optimizer = load_optimizer(model, config)
    best_validation_larp = 0
    nb_epochs = config["nb_epochs"]
    norm_loss = config["norm_loss"]
    top_k_loss = config.get("top_k_loss", None)
    top_k_scheduler = get_top_k_scheduler(config, nb_epochs)

    # ==== Scheduler ==== #
    scheduler = get_scheduler(config, train_loader)
    last_scheduler_update = 0

    if config["fine_tuning"]:
        checkpoint_path = osp.join(CHECKPOINT_FOLDER, config["checkpoint_name"])
        model, optimizer = load_checkpoint(model, optimizer, checkpoint_path)

    for e in range(nb_epochs):
        print("----- EPOCH {} -----".format(e + 1))
        top_k = update_top_k(top_k, top_k_scheduler, e)

        trainning_loss = train_epoch(
            train_loader,
            device,
            model,
            optimizer,
            scheduler,
            e,
            args.wandb,
            norm_loss,
            top_k_loss,
        )
        validation_loss, validation_lrap = validation_epoch(
            val_loader, device, model, norm_loss
        )
        last_scheduler_update += 1

        if args.wandb:
            wandb.log(
                {
                    "training_loss": trainning_loss,
                    "validation_loss": validation_loss,
                    "validation_lrap": validation_lrap,
                }
            )

        logging.info(
            f"Epoch {e + 1}: Training loss: {trainning_loss}, Validation loss: {validation_loss}, LRAP: {validation_lrap}"
        )

        if validation_lrap > best_validation_larp:
            best_validation_larp = validation_lrap
            save_path = osp.join(checkpoint_path, f"checkpoint_best.pt")
            save_checkpoint(model, optimizer, e, save_path)
            last_scheduler_update = 0

        if (
            config["scheduler"] == "exp_decay"
            and last_scheduler_update > config["wait_epochs"]
        ):
            scheduler = update_decay_scheduler(
                scheduler, config["scheduler_decay"], e * len(train_loader)
            )
            last_scheduler_update = 0

    save_path = osp.join(checkpoint_path, f"checkpoint_last.pt")
    save_checkpoint(model, optimizer, nb_epochs, save_path)

    logging.info(f"Best validation LARP: {best_validation_larp}")

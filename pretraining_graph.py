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

# Local application/library specific imports
from src.constants import *
from src.scheduler import cosineLR
from src.training import pretraining_graph
from src.utils import load_config, load_pretraining_model, load_optimizer, get_pretraining_dataloader, save_checkpoint, get_transform, get_scheduler

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

    # ==== Data augmentation ==== #
    transform, transform_params = get_transform(config)

    # ===== Dataloaders ===== #
    print("Loading datasets")
    loading_time = time.time()
    train_loader = get_pretraining_dataloader(config=config, transform=transform, transform_params=transform_params)
    print("Loading time: ", time.time() - loading_time)
    
    # ==== Device ==== #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==== Model ==== #
    student, teacher = load_pretraining_model(config)
    student = student.to(device)
    teacher = teacher.to(device)

    # ==== Optimizer ==== #
    optimizer = load_optimizer(student, config)
    start_epoch = 1
    nb_epochs = config["nb_epochs"]

    # ==== Scheduler ==== #
    scheduler = get_scheduler(config, train_loader)
    last_scheduler_update = 0
    momentum_teacher = cosineLR(nb_epochs, config["lr_teacher"], 1, len(train_loader))

    # ==== Center ==== #
    center = torch.zeros(config["nout"]).to(device)

    for e in range(start_epoch, start_epoch + nb_epochs):
        print("----- EPOCH {} -----".format(e))

        training_loss, center = pretraining_graph(
            train_loader,
            device,
            student,
            teacher,
            center,
            optimizer,
            scheduler,
            e,
            config["do_wandb"],
            config["momentum_center"],
            momentum_teacher,
            config["temperature_student"],
            config["temperature_teacher"]
        )

        if args.wandb:
            wandb.log(
                {
                    "training_loss_epoch": training_loss,
                }
            )

        logging.info(
            f"Epoch {e}: Training loss: {training_loss}"
        )

        save_checkpoint(
            student,
            optimizer,
            e,
            osp.join(checkpoint_path, f"checkpoint_{e}.pt"),
        )
    

# Related third-party imports
import torch
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# Local application/library specific imports
from src.constants import *
from src.loss import contrastive_loss


def train_epoch(train_loader, device, model, optimizer, do_wandb, norm_loss):
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

        graph_embeddings, text_embeddings = model(
            graph_batch, input_ids, attention_mask
        )
        loss = contrastive_loss(graph_embeddings, text_embeddings, normalize=norm_loss)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if do_wandb:
            wandb.log({"training_loss_step": loss.item()})

    average_loss = total_loss / len(train_loader)

    return average_loss


def label_ranking_average_precision(text_emeddings, graph_embeddings):
    similarities = cosine_similarity(text_emeddings, graph_embeddings)
    ground_truth = np.eye(similarities.shape[0])

    return label_ranking_average_precision_score(ground_truth, similarities)


def validation_epoch(val_loader, device, model, norm_loss):
    model.eval()
    total_loss = 0

    text_embeddings_list = []
    graph_embeddings_list = []
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

            graph_embeddings, text_embeddings = model(
                graph_batch, input_ids, attention_mask
            )
            loss = contrastive_loss(graph_embeddings, text_embeddings, normalize=norm_loss)
            total_loss += loss.item()

            text_embeddings_list.append(text_embeddings.cpu().numpy())
            graph_embeddings_list.append(graph_embeddings.cpu().numpy())

    average_loss = total_loss / len(val_loader)
    text_embeddings = np.concatenate(text_embeddings_list)
    graph_embeddings = np.concatenate(graph_embeddings_list)
    lrap = label_ranking_average_precision(text_embeddings, graph_embeddings)

    return average_loss, lrap

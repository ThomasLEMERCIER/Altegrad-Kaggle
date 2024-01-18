import torch
from tqdm import tqdm

from src.constants import *
from src.loss import contrastive_loss


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

        graph_embeddings, text_embeddings = model(
            graph_batch, input_ids, attention_mask
        )
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

            graph_embeddings, text_embeddings = model(
                graph_batch, input_ids, attention_mask
            )
            loss = contrastive_loss(graph_embeddings, text_embeddings)
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)

    return average_loss

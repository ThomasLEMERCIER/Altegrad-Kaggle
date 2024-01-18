import torch
from tqdm import tqdm

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import *


def text_inference(text_dataloader, device, text_model):
    text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(text_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            x_text = text_model(input_ids, attention_mask=attention_mask)
            text_embeddings.append(x_text.cpu().numpy())

    return np.concatenate(text_embeddings)


def graph_inference(graph_dataloader, device, graph_model):
    graph_embeddings = []

    with torch.no_grad():
        for batch in tqdm(graph_dataloader):
            x_graph = graph_model(batch.to(device))
            graph_embeddings.append(x_graph.cpu().numpy())

    return np.concatenate(graph_embeddings)


def text_graph_inference(text_graph_dataloader, device, model):
    text_model = model.get_text_encoder()
    graph_model = model.get_graph_encoder()

    text_embeddings = []
    graph_embeddings = []

    with torch.no_grad():
        for batch in tqdm(text_graph_dataloader):
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            batch.pop("input_ids")
            batch.pop("attention_mask")
            graph_batch = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            graph_batch = graph_batch.to(device)

            x_text = text_model(input_ids, attention_mask=attention_mask)
            x_graph = graph_model(graph_batch)

            text_embeddings.append(x_text.cpu().numpy())
            graph_embeddings.append(x_graph.cpu().numpy())

    return np.concatenate(text_embeddings), np.concatenate(graph_embeddings)


def label_ranking_average_precision(text_emeddings, graph_embeddings):
    similarities = cosine_similarity(text_emeddings, graph_embeddings)
    ground_truth = np.eye(similarities.shape[0])

    return label_ranking_average_precision_score(ground_truth, similarities)

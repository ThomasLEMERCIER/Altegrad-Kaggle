import torch
from tqdm import tqdm

import numpy as np

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

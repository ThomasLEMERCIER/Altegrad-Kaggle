# Related third-party imports
import torch
import wandb
import numpy as np
from tqdm import tqdm
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

# Local application/library specific imports
from src.constants import *
from src.loss import contrastive_loss, self_supervised_entropy


def train_epoch(
    train_loader, device, model, optimizer, scheduler, epoch, do_wandb, norm_loss
):
    model.train()
    total_loss = 0

    for it, batch in enumerate(tqdm(train_loader, desc="Training")):
        itx = it + len(train_loader) * epoch
        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler[itx]

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
            wandb.log({"training_loss_step": loss.item(), "lr": scheduler[itx]})

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
            loss = contrastive_loss(
                graph_embeddings, text_embeddings, normalize=norm_loss
            )
            total_loss += loss.item()

            text_embeddings_list.append(text_embeddings.cpu().numpy())
            graph_embeddings_list.append(graph_embeddings.cpu().numpy())

    average_loss = total_loss / len(val_loader)
    text_embeddings = np.concatenate(text_embeddings_list)
    graph_embeddings = np.concatenate(graph_embeddings_list)
    lrap = label_ranking_average_precision(text_embeddings, graph_embeddings)

    return average_loss, lrap

def pretraining_graph(train_loader, device, model_student, model_teacher, center, optimizer, scheduler, epoch, do_wandb, momentum_center, momentum_teacher, temperature_student, temperature_teacher):
    model_student.train()
    model_teacher.eval()
    total_loss = 0

    for it, batch in enumerate(tqdm(train_loader, desc="Training")):
        itx = it + len(train_loader) * epoch
        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler[itx]

        batch_u, batch_v = batch

        batch_u = batch_u.to(device)
        batch_v = batch_v.to(device)

        optimizer.zero_grad()

        student_u = model_student(batch_u)
        student_v = model_student(batch_v)

        with torch.no_grad():
            teacher_u = model_teacher(batch_u)
            teacher_v = model_teacher(batch_v)

        loss_uv = self_supervised_entropy(student_u, teacher_v, center, temperature_student, temperature_teacher)
        loss_vu = self_supervised_entropy(student_v, teacher_u, center, temperature_student, temperature_teacher)
        loss = loss_uv + loss_vu

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ema update for teacher model
        lr_teacher = momentum_teacher[itx]
        for param_s, param_t in zip(model_student.parameters(), model_teacher.parameters()):
            param_t.data = lr_teacher * param_t.data + (1 - lr_teacher) * param_s.detach().data

        if do_wandb:
            wandb.log({"training_loss_step": loss.item()})

        center = momentum_center * center + (1 - momentum_center) * torch.stack([student_u, teacher_v], dim=0).mean(dim=0)       

    average_loss = total_loss / len(train_loader)

    return average_loss, center

import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosineLR(epochs, warmup_epochs, eta_min, eta_max, loader_length):
    T_max = epochs * loader_length
    steps = np.arange(0, T_max)
    warmup_steps = warmup_epochs * loader_length

    lr = np.empty(T_max, dtype=np.float32)
    lr[:warmup_steps] = np.linspace(eta_min, eta_max, warmup_steps)
    lr[warmup_steps:] = eta_min + 0.5 * (eta_max - eta_min) * (
        1
        + np.cos((steps[warmup_steps:] - warmup_steps) / (T_max - warmup_steps) * np.pi)
    )

    return lr

def cosineLR(epochs, eta_min, eta_max, loader_length):
    T_max = epochs * loader_length
    steps = np.arange(0, T_max)

    lr = np.empty(T_max, dtype=np.float32)
    lr[:] = eta_min + 0.5 * (eta_max - eta_min) * (
        1
        + np.cos(steps / T_max * np.pi)
    )

    return lr

def constantLR(epochs, eta_min, loader_length):
    T_max = epochs * loader_length
    lr = np.ones(T_max, dtype=np.float32) * eta_min

    return lr

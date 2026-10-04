"""Explicit backend initialization belongs to commands, never package import."""

import os
import random

import numpy as np
import torch


def initialize(training):
    if os.environ.get("DDE_BACKEND", "pytorch") != "pytorch":
        raise ValueError("This reference implementation requires DDE_BACKEND=pytorch")
    os.environ["DDE_BACKEND"] = "pytorch"
    import deepxde as dde
    device = training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; sync the cuda extra and check the driver")
    torch.set_num_threads(training.cpu_threads)
    dde.config.set_default_float(training.dtype)
    torch.set_default_device(device)
    dde.config.set_random_seed(training.seed)
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    return dde, torch.device(device), getattr(torch, training.dtype)

"""Select the PyTorch backend only when an explicit operation begins."""

import os


def initialize(config):
    if os.environ.get("DDE_BACKEND", "pytorch") != "pytorch":
        raise ValueError("This project requires DDE_BACKEND=pytorch")
    os.environ["DDE_BACKEND"] = "pytorch"
    import torch
    import deepxde as dde
    device = config.t["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use the cuda extra and verify the driver")
    torch.set_num_threads(config.t["cpu_threads"])
    torch.set_default_device(device)
    dde.config.set_default_float(config.t["dtype"])
    dde.config.set_random_seed(config.t["seed"])
    return dde, torch.device(device), getattr(torch, config.t["dtype"])

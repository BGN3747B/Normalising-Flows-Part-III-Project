# flowlib/checkpointing.py

import torch

from .model import build_flow
from .scalers import StandardScaler, WhiteningScaler


def load_checkpoint(checkpoint_path, ndim, device, scaler_type="standard"):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    params = checkpoint["params"]

    flow = build_flow(
        ndim,
        num_layers=params.get("num_layers", 7),
        hidden_features=params.get("hidden_features", 64),
        num_blocks=params.get("num_blocks", 2),
        use_lu_linear=params.get("use_lu_linear", False)
    ).to(device)

    flow.load_state_dict(checkpoint["model_state_dict"])
    flow.eval()

    if scaler_type == "standard":
        scaler = StandardScaler()
        scaler.mean = checkpoint["scaler_mean"]
        scaler.std = checkpoint["scaler_std"]

    elif scaler_type == "zca":
        scaler = WhiteningScaler(mode="zca")
        scaler.mean = checkpoint["scaler_mean"]
        scaler.W = checkpoint["scaler_W"]
        scaler.W_inv = checkpoint["scaler_W_inv"]

    else:
        raise ValueError("scaler_type must be 'standard' or 'zca'")

    return flow, scaler, checkpoint
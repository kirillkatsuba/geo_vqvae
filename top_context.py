from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from .models import TopVQTransformer
from .models import TopPriorTransformer


def load_top_model(checkpoint_path: Path, device: torch.device) -> tuple[TopVQTransformer, list[str]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["model_config"]
    model = TopVQTransformer(
        input_dim=cfg["input_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        codebook_size=cfg["codebook_size"],
        dropout=cfg["dropout"],
        max_sequence_length=cfg["sequence_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, list(checkpoint["feature_columns"])


@torch.no_grad()
def encode_assay_embeddings(
    assays: pd.DataFrame,
    feature_columns: list[str],
    model: TopVQTransformer,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    codes = []
    embeddings = []
    for start in range(0, len(assays), batch_size):
        batch = assays.iloc[start : start + batch_size][feature_columns].to_numpy(dtype=np.float32)
        x = torch.tensor(batch, device=device).unsqueeze(0)
        mask = torch.ones(1, x.size(1), dtype=torch.bool, device=device)
        out_codes, z_q = model.encode(x, mask=mask)
        codes.append(out_codes.squeeze(0).detach().cpu().numpy())
        embeddings.append(z_q.squeeze(0).detach().cpu().numpy())
    return np.concatenate(codes), np.concatenate(embeddings)


def attach_top_context(
    blocks: pd.DataFrame,
    assays: pd.DataFrame,
    assay_embeddings: np.ndarray,
    k: int = 8,
    prefix: str = "top",
) -> tuple[pd.DataFrame, list[str]]:
    coords = ["X", "Y", "Z"]
    n_neighbors = min(k, len(assays))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(assays[coords].to_numpy(dtype=float))
    distances, indices = nn.kneighbors(blocks[coords].to_numpy(dtype=float))
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    context = np.einsum("nk,nkd->nd", weights, assay_embeddings[indices])
    out = blocks.copy()
    columns = [f"{prefix}_{idx}" for idx in range(context.shape[1])]
    for idx, col in enumerate(columns):
        out[col] = context[:, idx].astype(np.float32)
    out[f"{prefix}_nearest_distance"] = distances[:, 0].astype(np.float32)
    return out, columns


def load_top_prior(checkpoint_path: Path, device: torch.device) -> tuple[TopPriorTransformer, dict]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["model_config"]
    model = TopPriorTransformer(
        block_dim=cfg["block_dim"],
        codebook_size=cfg["codebook_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        max_sequence_length=cfg["sequence_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def attach_prior_top_context(
    blocks: pd.DataFrame,
    prior: TopPriorTransformer,
    top_model: TopVQTransformer,
    block_feature_columns: list[str],
    sequence_length: int,
    device: torch.device,
    prefix: str = "top",
) -> tuple[pd.DataFrame, list[str]]:
    from .dataset import chunk_indices, order_by_xyz

    out = blocks.copy()
    order = order_by_xyz(out)
    sequences = chunk_indices(order, sequence_length)
    context = np.full((len(out), top_model.d_model), np.nan, dtype=np.float32)
    code_values = np.full(len(out), -1, dtype=np.int64)
    for seq in sequences:
        block = torch.tensor(
            out.iloc[seq][block_feature_columns].to_numpy(dtype=np.float32),
            device=device,
        ).unsqueeze(0)
        mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
        codes = prior.generate(block, mask=mask).squeeze(0)
        z = top_model.quantizer.embedding(codes)
        context[seq] = z.detach().cpu().numpy()
        code_values[seq] = codes.detach().cpu().numpy()
    columns = [f"{prefix}_{idx}" for idx in range(context.shape[1])]
    for idx, col in enumerate(columns):
        out[col] = context[:, idx]
    out[f"{prefix}_prior_code"] = code_values
    return out, columns

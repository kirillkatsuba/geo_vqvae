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
    columns = [f"{prefix}_{idx}" for idx in range(context.shape[1])]
    context_df = pd.DataFrame(context.astype(np.float32), columns=columns, index=blocks.index)
    distance_df = pd.DataFrame(
        {f"{prefix}_nearest_distance": distances[:, 0].astype(np.float32)},
        index=blocks.index,
    )
    out = pd.concat([blocks.copy(), context_df, distance_df], axis=1)
    return out, columns


def attach_nearest_top_code(blocks: pd.DataFrame, assays: pd.DataFrame, assay_codes: np.ndarray) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(assays[["X", "Y", "Z"]].to_numpy(dtype=float))
    distance, index = nn.kneighbors(blocks[["X", "Y", "Z"]].to_numpy(dtype=float))
    out = blocks.copy()
    out["top_code_label"] = assay_codes[index[:, 0]].astype(np.int64)
    out["top_code_label_distance"] = distance[:, 0].astype(np.float32)
    return out


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

    order = order_by_xyz(blocks)
    sequences = chunk_indices(order, sequence_length)
    context = np.full((len(blocks), top_model.d_model), np.nan, dtype=np.float32)
    code_values = np.full(len(blocks), -1, dtype=np.int64)
    for seq in sequences:
        block = torch.tensor(
            blocks.iloc[seq][block_feature_columns].to_numpy(dtype=np.float32),
            device=device,
        ).unsqueeze(0)
        mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
        codes = prior.generate(block, mask=mask).squeeze(0)
        z = top_model.quantizer.embedding(codes)
        context[seq] = z.detach().cpu().numpy()
        code_values[seq] = codes.detach().cpu().numpy()
    columns = [f"{prefix}_{idx}" for idx in range(context.shape[1])]
    context_df = pd.DataFrame(context, columns=columns, index=blocks.index)
    code_df = pd.DataFrame({f"{prefix}_prior_code": code_values}, index=blocks.index)
    out = pd.concat([blocks.copy(), context_df, code_df], axis=1)
    return out, columns


@torch.no_grad()
def attach_prior_top_context_warm_start(
    blocks: pd.DataFrame,
    context_blocks: pd.DataFrame,
    context_codes: np.ndarray,
    prior: TopPriorTransformer,
    top_model: TopVQTransformer,
    block_feature_columns: list[str],
    sequence_length: int,
    warm_start_length: int,
    device: torch.device,
    prefix: str = "top",
) -> tuple[pd.DataFrame, list[str]]:
    from .dataset import order_by_xyz

    blocks = blocks.reset_index(drop=True)
    context_blocks = context_blocks.reset_index(drop=True)
    order = order_by_xyz(blocks)
    context_order = order_by_xyz(context_blocks)
    warm_start_length = max(0, int(warm_start_length))
    sequence_length = max(2, int(sequence_length))

    if warm_start_length <= 0 or len(context_blocks) == 0:
        return attach_prior_top_context(
            blocks,
            prior=prior,
            top_model=top_model,
            block_feature_columns=block_feature_columns,
            sequence_length=sequence_length,
            device=device,
            prefix=prefix,
        )

    context_tail = context_order[-warm_start_length:]
    history_features = context_blocks.iloc[context_tail][block_feature_columns].reset_index(drop=True)
    history_codes = np.asarray(context_codes, dtype=np.int64)[context_tail].copy()

    context = np.full((len(blocks), top_model.d_model), np.nan, dtype=np.float32)
    code_values = np.full(len(blocks), -1, dtype=np.int64)

    cursor = 0
    while cursor < len(order):
        prefix_len = min(len(history_codes), warm_start_length, sequence_length - 1)
        current_len = min(sequence_length - prefix_len, len(order) - cursor)
        seq = order[cursor : cursor + current_len]

        prefix_frame = history_features.tail(prefix_len)
        current_frame = blocks.iloc[seq][block_feature_columns]
        combined = pd.concat([prefix_frame, current_frame], ignore_index=True)
        block = torch.tensor(combined.to_numpy(dtype=np.float32), device=device).unsqueeze(0)
        mask = torch.ones(1, len(combined), dtype=torch.bool, device=device)
        prefix_tensor = torch.tensor(history_codes[-prefix_len:], dtype=torch.long, device=device).unsqueeze(0)
        codes = prior.generate_with_prefix(block, prefix_tensor, mask=mask).squeeze(0)
        current_codes = codes[prefix_len:]
        z = top_model.quantizer.embedding(current_codes)

        context[seq] = z.detach().cpu().numpy()
        code_values[seq] = current_codes.detach().cpu().numpy()

        history_features = pd.concat(
            [history_features, current_frame.reset_index(drop=True)],
            ignore_index=True,
        ).tail(warm_start_length)
        history_codes = np.concatenate([history_codes, code_values[seq]])[-warm_start_length:]
        cursor += current_len

    columns = [f"{prefix}_{idx}" for idx in range(context.shape[1])]
    context_df = pd.DataFrame(context, columns=columns, index=blocks.index)
    code_df = pd.DataFrame({f"{prefix}_prior_code": code_values}, index=blocks.index)
    out = pd.concat([blocks.copy(), context_df, code_df], axis=1)
    return out, columns

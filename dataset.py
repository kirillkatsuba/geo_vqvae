from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def chunk_indices(order: np.ndarray, sequence_length: int) -> list[np.ndarray]:
    return [
        order[start : start + sequence_length]
        for start in range(0, len(order), sequence_length)
        if len(order[start : start + sequence_length]) > 1
    ]


def order_by_xyz(df: pd.DataFrame) -> np.ndarray:
    cols = [col for col in ["domain", "Y", "X", "Z"] if col in df.columns]
    return df.sort_values(cols).index.to_numpy()


class FeatureSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        sequences: Sequence[Sequence[int]],
    ):
        self.df = df.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.sequences = [np.asarray(seq, dtype=np.int64) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        values = self.df.iloc[seq][self.feature_columns].to_numpy(dtype=np.float32)
        return {"features": torch.tensor(values), "order": torch.tensor(seq, dtype=torch.long)}


class LowSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        block_feature_columns: Sequence[str],
        target_columns: Sequence[str],
        top_context_columns: Sequence[str],
        sequences: Sequence[Sequence[int]],
    ):
        self.df = df.reset_index(drop=True)
        self.block_feature_columns = list(block_feature_columns)
        self.target_columns = list(target_columns)
        self.top_context_columns = list(top_context_columns)
        self.sequences = [np.asarray(seq, dtype=np.int64) for seq in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        frame = self.df.iloc[seq]
        return {
            "block_features": torch.tensor(frame[self.block_feature_columns].to_numpy(dtype=np.float32)),
            "top_context": torch.tensor(frame[self.top_context_columns].to_numpy(dtype=np.float32)),
            "targets": torch.tensor(frame[self.target_columns].to_numpy(dtype=np.float32)),
            "order": torch.tensor(seq, dtype=torch.long),
        }


def collate_padded(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(next(iter(item.values())).shape[0] for item in items)
    batch: dict[str, torch.Tensor] = {}
    for key in items[0].keys():
        if key == "order":
            out = torch.full((len(items), max_len), -1, dtype=torch.long)
            for idx, item in enumerate(items):
                out[idx, : item[key].shape[0]] = item[key]
            batch[key] = out
        else:
            dim = items[0][key].shape[-1]
            out = torch.zeros(len(items), max_len, dim, dtype=items[0][key].dtype)
            for idx, item in enumerate(items):
                out[idx, : item[key].shape[0]] = item[key]
            batch[key] = out
    mask = torch.zeros(len(items), max_len, dtype=torch.bool)
    if "features" in items[0]:
        first_key = "features"
    elif "targets" in items[0]:
        first_key = "targets"
    else:
        first_key = "block_features"
    for idx, item in enumerate(items):
        mask[idx, : item[first_key].shape[0]] = True
    batch["mask"] = mask
    return batch

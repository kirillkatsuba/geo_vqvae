from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CorrelationReference:
    name: str
    matrix: np.ndarray


def _finite_array(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return values[np.isfinite(values).all(axis=1)]


def correlation_matrix(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dim = values.shape[1]
    if values.shape[0] < dim + 2:
        return np.eye(dim, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=0, keepdims=True)
    z = centered / np.maximum(std, eps)
    corr = z.T @ z / max(1, z.shape[0] - 1)
    return nearest_correlation_matrix(corr)


def nearest_correlation_matrix(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    psd = (eigvecs * eigvals) @ eigvecs.T
    diag = np.sqrt(np.maximum(np.diag(psd), eps))
    corr = psd / np.outer(diag, diag)
    np.fill_diagonal(corr, 1.0)
    return corr


def _matrix_power(matrix: np.ndarray, power: float, eps: float = 1e-6) -> np.ndarray:
    matrix = nearest_correlation_matrix(matrix, eps=eps)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * np.power(eigvals, power)) @ eigvecs.T


def _block_reference_values(prepared_dir: Path, columns: list[str], split: str) -> np.ndarray:
    if split == "north_known":
        north = pd.read_parquet(prepared_dir / "north_blocks.parquet")
        frame = north.loc[north["split"] == "test_north_known"]
        return _finite_array(frame, columns)

    center = pd.read_parquet(prepared_dir / "center_blocks.parquet")
    if split == "train":
        frame = center.loc[center["split"] == "train"]
    elif split == "train_val":
        frame = center.loc[center["split"].isin(["train", "val"])]
    elif split == "center_all":
        frame = center.loc[center["has_targets"]]
    else:
        raise ValueError(f"Unknown block reference split: {split}")
    return _finite_array(frame, columns)


def build_correlation_reference(
    prepared_dir: Path,
    columns: list[str],
    mode: str,
    block_split: str = "train",
    block_weight: float = 1.0,
    assay_weight: float = 1.0,
) -> CorrelationReference | None:
    if mode == "none":
        return None

    matrices: list[np.ndarray] = []
    weights: list[float] = []
    names: list[str] = []

    if mode in {"block", "blend"} and block_weight > 0:
        values = _block_reference_values(prepared_dir, columns, block_split)
        if values.size:
            matrices.append(correlation_matrix(values))
            weights.append(float(block_weight))
            names.append(f"block:{block_split}")

    if mode in {"assay", "blend"} and assay_weight > 0:
        assays = pd.read_parquet(prepared_dir / "assays.parquet")
        if set(columns).issubset(assays.columns):
            frame = assays.loc[assays["has_targets"]]
            values = _finite_array(frame, columns)
            if values.size:
                matrices.append(correlation_matrix(values))
                weights.append(float(assay_weight))
                names.append("assay")

    if not matrices:
        raise ValueError(
            f"Could not build correlation reference mode={mode!r}. "
            "Check that prepared data contains all target columns."
        )

    weight_sum = max(sum(weights), 1e-8)
    blended = sum(weight * matrix for weight, matrix in zip(weights, matrices)) / weight_sum
    return CorrelationReference(name="+".join(names), matrix=nearest_correlation_matrix(blended))


def match_prediction_correlation(
    predictions: np.ndarray,
    reference_corr: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Affine post-calibration of prediction correlations.

    The transform whitens the current prediction correlation matrix and recolors
    it with the reference matrix. Per-target mean and standard deviation are
    restored afterwards, so this changes mostly cross-target dependency rather
    than the marginal scale.
    """

    out = np.asarray(predictions, dtype=np.float64).copy()
    finite = np.isfinite(out).all(axis=1)
    values = out[finite]
    dim = out.shape[1]
    if values.shape[0] < dim + 2:
        return out.astype(np.float32)

    strength = float(np.clip(strength, 0.0, 1.0))
    mean = values.mean(axis=0, keepdims=True)
    std = np.maximum(values.std(axis=0, ddof=0, keepdims=True), 1e-8)
    z = (values - mean) / std

    source_corr = correlation_matrix(z)
    target_corr = nearest_correlation_matrix(reference_corr)
    transform = _matrix_power(source_corr, -0.5) @ _matrix_power(target_corr, 0.5)
    matched = z @ transform
    matched = (matched - matched.mean(axis=0, keepdims=True)) / np.maximum(
        matched.std(axis=0, ddof=0, keepdims=True),
        1e-8,
    )

    adjusted_z = z + strength * (matched - z)
    adjusted_z = (adjusted_z - adjusted_z.mean(axis=0, keepdims=True)) / np.maximum(
        adjusted_z.std(axis=0, ddof=0, keepdims=True),
        1e-8,
    )
    out[finite] = adjusted_z * std + mean
    return out.astype(np.float32)


def apply_correlation_adjustment_to_frame(
    frame: pd.DataFrame,
    target_columns: list[str],
    reference: CorrelationReference,
    strength: float,
    keep_unadjusted: bool = False,
) -> pd.DataFrame:
    out = frame.copy()
    pred_columns = [f"pred_{target}" for target in target_columns]
    values = out[pred_columns].to_numpy(dtype=np.float64)
    adjusted = match_prediction_correlation(values, reference.matrix, strength=strength)

    if keep_unadjusted:
        for target in target_columns:
            out[f"pred_unadjusted_{target}"] = out[f"pred_{target}"]

    for idx, target in enumerate(target_columns):
        out[f"pred_{target}"] = adjusted[:, idx]
        true_col = f"true_{target}"
        if true_col in out.columns:
            out[f"error_{target}"] = out[f"pred_{target}"] - out[true_col]
    return out

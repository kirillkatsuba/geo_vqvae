from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import LowSequenceDataset, chunk_indices, collate_padded, order_by_xyz
from .models import LowVQVAE2, code_ce_loss, correlation_loss, masked_mse, reference_correlation_loss
from .preprocessing import TargetScaler
from .top_context import (
    attach_prior_top_context,
    attach_top_context,
    encode_assay_embeddings,
    load_top_model,
    load_top_prior,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class LowConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    codebook_size: int = 256
    dropout: float = 0.1
    sequence_length: int = 1024
    top_k: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train low-level block VQ-VAE2 conditioned on top assay codes.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared"))
    parser.add_argument("--top-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--top-prior-checkpoint",
        type=Path,
        default=None,
        help="Optional autoregressive top-code prior. If set, low model uses generated top context instead of nearest-assay context.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/runs/low_v1"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
    parser.add_argument("--lambda-code", type=float, default=0.2)
    parser.add_argument("--lambda-corr", type=float, default=0.1)
    parser.add_argument(
        "--corr-mode",
        choices=["none", "batch", "global", "batch_global"],
        default="batch",
        help=(
            "Correlation regularization mode. batch matches each prediction batch to its targets; "
            "global matches predictions to reference correlations from source block model and assays."
        ),
    )
    parser.add_argument(
        "--corr-reference-split",
        choices=["train", "train_val"],
        default="train",
        help="Block-model split used to build the global block correlation reference.",
    )
    parser.add_argument("--corr-block-weight", type=float, default=1.0)
    parser.add_argument("--corr-assay-weight", type=float, default=1.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def numpy_correlation_matrix(values: pd.DataFrame) -> np.ndarray:
    arr = values.dropna().to_numpy(dtype=np.float32)
    dim = values.shape[1]
    if arr.shape[0] < dim + 2:
        return np.eye(dim, dtype=np.float32)
    arr = arr - arr.mean(axis=0, keepdims=True)
    arr = arr / np.maximum(arr.std(axis=0, keepdims=True), 1e-6)
    return (arr.T @ arr / max(1, arr.shape[0] - 1)).astype(np.float32)


def build_correlation_references(
    prepared_dir: Path,
    center: pd.DataFrame,
    assays: pd.DataFrame,
    metadata: dict,
    split: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    target_columns = list(metadata["target_columns"])
    scaled_target_columns = list(metadata["scaled_target_columns"])
    block_splits = ["train"] if split == "train" else ["train", "val"]
    block_ref = center.loc[center["split"].isin(block_splits), scaled_target_columns]
    refs = {
        "block": torch.tensor(numpy_correlation_matrix(block_ref), device=device),
    }

    available_targets = [col for col in target_columns if col in assays.columns]
    if len(available_targets) == len(target_columns):
        assay_ref = assays.loc[assays["has_targets"], target_columns].dropna()
        if not assay_ref.empty:
            scaler = TargetScaler.load(prepared_dir / "target_scaler.json")
            assay_ref_scaled = scaler.transform(assay_ref)
            refs["assay"] = torch.tensor(numpy_correlation_matrix(assay_ref_scaled), device=device)
    return refs


def correlation_regularization(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    references: dict[str, torch.Tensor],
    block_weight: float,
    assay_weight: float,
) -> torch.Tensor:
    if mode == "none":
        return pred.new_tensor(0.0)

    losses = []
    if mode in {"batch", "batch_global"}:
        losses.append(correlation_loss(pred, target, mask))

    if mode in {"global", "batch_global"}:
        weighted = []
        weights = []
        if block_weight > 0 and "block" in references:
            weighted.append(block_weight * reference_correlation_loss(pred, references["block"], mask))
            weights.append(block_weight)
        if assay_weight > 0 and "assay" in references:
            weighted.append(assay_weight * reference_correlation_loss(pred, references["assay"], mask))
            weights.append(assay_weight)
        if weighted:
            losses.append(sum(weighted) / max(sum(weights), 1e-6))

    if not losses:
        return pred.new_tensor(0.0)
    return sum(losses) / len(losses)


def evaluate(
    model: LowVQVAE2,
    loader: DataLoader,
    device: torch.device,
    lambda_corr: float,
    corr_mode: str,
    corr_references: dict[str, torch.Tensor],
    corr_block_weight: float,
    corr_assay_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    recons = []
    corrs = []
    with torch.no_grad():
        for batch in loader:
            block = batch["block_features"].to(device)
            top = batch["top_context"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            pred, _ = model.generate(block, top, mask=mask)
            recon = masked_mse(pred, targets, mask)
            corr = correlation_regularization(
                pred,
                targets,
                mask,
                corr_mode,
                corr_references,
                corr_block_weight,
                corr_assay_weight,
            )
            loss = recon + lambda_corr * corr
            losses.append(float(loss.detach().cpu()))
            recons.append(float(recon.detach().cpu()))
            corrs.append(float(corr.detach().cpu()))
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_recon": float(np.mean(recons)) if recons else float("nan"),
        "val_corr": float(np.mean(corrs)) if corrs else float("nan"),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    metadata = json.loads((args.prepared_dir / "metadata.json").read_text())
    assays = pd.read_parquet(args.prepared_dir / "assays.parquet")
    center = pd.read_parquet(args.prepared_dir / "center_blocks.parquet")

    top_model, top_feature_columns = load_top_model(args.top_checkpoint, device)
    center_train_val = center.loc[center["split"].isin(["train", "val"])].reset_index(drop=True)
    if args.top_prior_checkpoint is not None:
        top_prior, top_prior_ckpt = load_top_prior(args.top_prior_checkpoint, device)
        center_ctx, top_columns = attach_prior_top_context(
            center_train_val,
            prior=top_prior,
            top_model=top_model,
            block_feature_columns=metadata["block_feature_columns"],
            sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
            device=device,
        )
    else:
        _, assay_embeddings = encode_assay_embeddings(assays, top_feature_columns, top_model, device)
        center_ctx, top_columns = attach_top_context(center_train_val, assays, assay_embeddings, k=args.top_k)

    corr_references = build_correlation_references(
        prepared_dir=args.prepared_dir,
        center=center,
        assays=assays,
        metadata=metadata,
        split=args.corr_reference_split,
        device=device,
    )

    train_df = center_ctx.loc[center_ctx["split"] == "train"].reset_index(drop=True)
    val_df = center_ctx.loc[center_ctx["split"] == "val"].reset_index(drop=True)
    block_columns = metadata["block_feature_columns"]
    target_columns = metadata["scaled_target_columns"]

    train_sequences = chunk_indices(order_by_xyz(train_df), args.sequence_length)
    val_sequences = chunk_indices(order_by_xyz(val_df), args.sequence_length)
    if args.max_sequences > 0:
        train_sequences = train_sequences[: args.max_sequences]
        val_sequences = val_sequences[: max(1, min(len(val_sequences), args.max_sequences))]

    train_ds = LowSequenceDataset(train_df, block_columns, target_columns, top_columns, train_sequences)
    val_ds = LowSequenceDataset(val_df, block_columns, target_columns, top_columns, val_sequences)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_padded)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_padded)

    config = LowConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        top_k=args.top_k,
    )
    model = LowVQVAE2(
        block_dim=len(block_columns),
        target_dim=len(target_columns),
        top_dim=len(top_columns),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        codebook_size=config.codebook_size,
        dropout=config.dropout,
        max_sequence_length=config.sequence_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    rows = []
    best = float("inf")
    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None and not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc="low epochs")
    for epoch in epoch_iter:
        model.train()
        train_losses = []
        train_recon = []
        train_corr = []
        batch_iter = train_loader
        if tqdm is not None and not args.no_progress:
            batch_iter = tqdm(train_loader, desc=f"low {epoch}/{args.epochs}", leave=False)
        for batch in batch_iter:
            block = batch["block_features"].to(device)
            top = batch["top_context"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(block, top, targets, mask=mask)
            recon = masked_mse(out["recon"], targets, mask)
            ce = code_ce_loss(out["logits"], out["codes"].detach(), mask)
            corr = correlation_regularization(
                out["recon"],
                targets,
                mask,
                args.corr_mode,
                corr_references,
                args.corr_block_weight,
                args.corr_assay_weight,
            )
            loss = recon + args.lambda_vq * out["vq_loss"] + args.lambda_code * ce + args.lambda_corr * corr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_recon.append(float(recon.detach().cpu()))
            train_corr.append(float(corr.detach().cpu()))

        val = evaluate(
            model,
            val_loader,
            device,
            args.lambda_corr,
            args.corr_mode,
            corr_references,
            args.corr_block_weight,
            args.corr_assay_weight,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_recon": float(np.mean(train_recon)),
            "train_corr": float(np.mean(train_corr)),
            **val,
        }
        rows.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_loss={row['val_loss']:.6f} val_recon={row['val_recon']:.6f} val_corr={row['val_corr']:.6f}"
        )
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "block_dim": len(block_columns),
                        "target_dim": len(target_columns),
                        "top_dim": len(top_columns),
                        **asdict(config),
                        "corr_mode": args.corr_mode,
                        "corr_reference_split": args.corr_reference_split,
                        "corr_block_weight": args.corr_block_weight,
                        "corr_assay_weight": args.corr_assay_weight,
                    },
                    "block_feature_columns": block_columns,
                    "target_columns": target_columns,
                    "top_context_columns": top_columns,
                    "metadata": metadata,
                    "top_checkpoint": str(args.top_checkpoint),
                    "top_prior_checkpoint": str(args.top_prior_checkpoint) if args.top_prior_checkpoint is not None else "",
                    "epoch": epoch,
                    "val_loss": row["val_loss"],
                },
                args.output_dir / "best_low.pt",
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Saved low checkpoint: {args.output_dir / 'best_low.pt'}")


if __name__ == "__main__":
    main()

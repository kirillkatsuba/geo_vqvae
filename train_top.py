from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import FeatureSequenceDataset, chunk_indices, collate_padded, order_by_xyz
from .models import TopVQTransformer, masked_mse

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class TopConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    codebook_size: int = 256
    dropout: float = 0.1
    sequence_length: int = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train top-level VQ Transformer on assay tokens.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/runs/top_v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-dropout", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--lambda-vq", type=float, default=1.0)
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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    metadata = json.loads((args.prepared_dir / "metadata.json").read_text())
    feature_columns = metadata["assay_feature_columns"]
    assays = pd.read_parquet(args.prepared_dir / "assays.parquet")
    order = order_by_xyz(assays)
    sequences = chunk_indices(order, args.sequence_length)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    dataset = FeatureSequenceDataset(assays, feature_columns, sequences)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_padded)

    config = TopConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
    )
    model = TopVQTransformer(
        input_dim=len(feature_columns),
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
        epoch_iter = tqdm(epoch_iter, desc="top epochs")
    for epoch in epoch_iter:
        losses = []
        perplexities = []
        batch_iter = loader
        if tqdm is not None and not args.no_progress:
            batch_iter = tqdm(loader, desc=f"top {epoch}/{args.epochs}", leave=False)
        for batch in batch_iter:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            model_input = features
            if args.feature_dropout > 0:
                keep = torch.rand_like(model_input) > args.feature_dropout
                model_input = model_input * keep.to(model_input.dtype)
            if args.input_noise_std > 0:
                model_input = model_input + args.input_noise_std * torch.randn_like(model_input)
            optimizer.zero_grad(set_to_none=True)
            out = model(model_input, mask=mask)
            recon = masked_mse(out["recon"], features, mask)
            loss = recon + args.lambda_vq * out["vq_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            perplexities.append(float(out["perplexity"].detach().cpu()))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "perplexity": float(np.mean(perplexities))}
        rows.append(row)
        print(f"epoch={epoch} loss={row['loss']:.6f} perplexity={row['perplexity']:.2f}")
        if row["loss"] < best:
            best = row["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_dim": len(feature_columns),
                        **asdict(config),
                    },
                    "feature_columns": feature_columns,
                    "metadata": metadata,
                    "epoch": epoch,
                    "loss": row["loss"],
                },
                args.output_dir / "best_top.pt",
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Saved top checkpoint: {args.output_dir / 'best_top.pt'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

from .dataset import chunk_indices, collate_padded, order_by_xyz
from .models import TopPriorTransformer, code_ce_loss, shift_codes_right
from .top_context import encode_assay_embeddings, load_top_model

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class TopPriorConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    codebook_size: int = 256
    dropout: float = 0.1
    sequence_length: int = 1024


class TopPriorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_columns: list[str], code_column: str, sequences: list[np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.code_column = code_column
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        frame = self.df.iloc[seq]
        return {
            "block_features": torch.tensor(frame[self.feature_columns].to_numpy(dtype=np.float32)),
            "codes": torch.tensor(frame[self.code_column].to_numpy(dtype=np.int64), dtype=torch.long),
            "order": torch.tensor(seq, dtype=torch.long),
        }


def collate_prior(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = collate_padded([{k: v for k, v in item.items() if k != "codes"} for item in items])
    max_len = batch["mask"].size(1)
    codes = torch.zeros(len(items), max_len, dtype=torch.long)
    for idx, item in enumerate(items):
        codes[idx, : item["codes"].shape[0]] = item["codes"]
    batch["codes"] = codes
    return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train autoregressive top-code prior on block tokens.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared"))
    parser.add_argument("--top-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/runs/top_prior_v1"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
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


def attach_nearest_top_code(blocks: pd.DataFrame, assays: pd.DataFrame, assay_codes: np.ndarray) -> pd.DataFrame:
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(assays[["X", "Y", "Z"]].to_numpy(dtype=float))
    distance, index = nn.kneighbors(blocks[["X", "Y", "Z"]].to_numpy(dtype=float))
    out = blocks.copy()
    out["top_code_label"] = assay_codes[index[:, 0]].astype(np.int64)
    out["top_code_label_distance"] = distance[:, 0].astype(np.float32)
    return out


def evaluate(model: TopPriorTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            block = batch["block_features"].to(device)
            codes = batch["codes"].to(device)
            mask = batch["mask"].to(device)
            prev = shift_codes_right(codes, model.bos_code)
            logits = model(block, prev, mask=mask)
            loss = code_ce_loss(logits, codes, mask)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    metadata = json.loads((args.prepared_dir / "metadata.json").read_text())
    block_columns = metadata["block_feature_columns"]
    assays = pd.read_parquet(args.prepared_dir / "assays.parquet")
    center = pd.read_parquet(args.prepared_dir / "center_blocks.parquet")
    top_model, top_feature_columns = load_top_model(args.top_checkpoint, device)
    assay_codes, _ = encode_assay_embeddings(assays, top_feature_columns, top_model, device)
    center = attach_nearest_top_code(center, assays, assay_codes)

    train_df = center.loc[center["split"] == "train"].reset_index(drop=True)
    val_df = center.loc[center["split"] == "val"].reset_index(drop=True)
    train_sequences = chunk_indices(order_by_xyz(train_df), args.sequence_length)
    val_sequences = chunk_indices(order_by_xyz(val_df), args.sequence_length)
    if args.max_sequences > 0:
        train_sequences = train_sequences[: args.max_sequences]
        val_sequences = val_sequences[: max(1, min(len(val_sequences), args.max_sequences))]

    train_loader = DataLoader(
        TopPriorDataset(train_df, block_columns, "top_code_label", train_sequences),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_prior,
    )
    val_loader = DataLoader(
        TopPriorDataset(val_df, block_columns, "top_code_label", val_sequences),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_prior,
    )

    config = TopPriorConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        codebook_size=top_model.quantizer.codebook_size,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
    )
    model = TopPriorTransformer(
        block_dim=len(block_columns),
        codebook_size=config.codebook_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_sequence_length=config.sequence_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    rows = []
    best = float("inf")
    epoch_iter = range(1, args.epochs + 1)
    if tqdm is not None and not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc="top-prior epochs")
    for epoch in epoch_iter:
        model.train()
        losses = []
        batch_iter = train_loader
        if tqdm is not None and not args.no_progress:
            batch_iter = tqdm(train_loader, desc=f"top-prior {epoch}/{args.epochs}", leave=False)
        for batch in batch_iter:
            block = batch["block_features"].to(device)
            codes = batch["codes"].to(device)
            mask = batch["mask"].to(device)
            prev = shift_codes_right(codes, model.bos_code)
            optimizer.zero_grad(set_to_none=True)
            logits = model(block, prev, mask=mask)
            loss = code_ce_loss(logits, codes, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = evaluate(model, val_loader, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "block_dim": len(block_columns),
                        **asdict(config),
                    },
                    "block_feature_columns": block_columns,
                    "top_checkpoint": str(args.top_checkpoint),
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                args.output_dir / "best_top_prior.pt",
            )

    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Saved top prior checkpoint: {args.output_dir / 'best_top_prior.pt'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-geo-vqvae")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import LowSequenceDataset, chunk_indices, collate_padded, order_by_xyz
from .models import LowVQVAE2
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-level VQ-VAE2 on center/north known blocks.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared"))
    parser.add_argument("--low-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/eval/low_v1"))
    parser.add_argument("--domain", choices=["center_val", "north_known", "both"], default="both")
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-plot-points", type=int, default=60000)
    parser.add_argument("--decode-mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-plots", action="store_true")
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


def load_low_model(path: Path, device: torch.device) -> tuple[LowVQVAE2, dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt["model_config"]
    model = LowVQVAE2(
        block_dim=cfg["block_dim"],
        target_dim=cfg["target_dim"],
        top_dim=cfg["top_dim"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        codebook_size=cfg["codebook_size"],
        dropout=cfg["dropout"],
        max_sequence_length=cfg["sequence_length"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def compute_metrics(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        y = df[f"true_{target}"].to_numpy(dtype=float)
        p = df[f"pred_{target}"].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(p)
        y = y[mask]
        p = p[mask]
        err = p - y
        denom = np.sum((y - y.mean()) ** 2)
        rows.append(
            {
                "target": target,
                "n": int(len(y)),
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "R2": float(1.0 - np.sum(err**2) / denom) if denom > 1e-12 else np.nan,
                "bias": float(np.mean(err)),
                "true_mean": float(np.mean(y)),
                "pred_mean": float(np.mean(p)),
            }
        )
    return pd.DataFrame(rows)


def plot_xy_maps(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: list[str],
    output_dir: Path,
    max_points: int,
    seed: int = 42,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(df) > max_points:
        plot_df = df.iloc[rng.choice(len(df), size=max_points, replace=False)].copy()
    else:
        plot_df = df.copy()

    for target in targets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        columns = [f"true_{target}", f"pred_{target}", f"error_{target}"]
        titles = [f"{target} true", f"{target} pred", f"{target} error"]
        for ax, col, title in zip(axes, columns, titles):
            values = plot_df[col].to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not np.any(finite):
                ax.set_title(title)
                continue
            if col.startswith("error_"):
                vmax = float(np.nanpercentile(np.abs(values), 98))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                vmin = -vmax
                cmap = "coolwarm"
            else:
                vmin = float(np.nanpercentile(values, 2))
                vmax = float(np.nanpercentile(values, 98))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin = float(np.nanmin(values))
                    vmax = float(np.nanmax(values))
                cmap = "viridis"
            sc = ax.scatter(
                plot_df["X"],
                plot_df["Y"],
                c=values,
                s=2,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidths=0,
            )
            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        metric_row = metrics.loc[metrics["target"] == target]
        if not metric_row.empty:
            row = metric_row.iloc[0]
            fig.suptitle(
                f"{target}: MAE={row['MAE']:.4g}, RMSE={row['RMSE']:.4g}, R2={row['R2']:.4g}",
                fontsize=12,
            )
        fig.savefig(plot_dir / f"xy_{target}.png", dpi=180)
        plt.close(fig)


@torch.no_grad()
def predict_domain(
    name: str,
    blocks: pd.DataFrame,
    model: LowVQVAE2,
    ckpt: dict,
    target_scaler: TargetScaler,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool,
    decode_mode: str = "hard",
    softmax_temperature: float = 1.0,
) -> pd.DataFrame:
    block_columns = ckpt["block_feature_columns"]
    target_columns = ckpt["target_columns"]
    top_columns = ckpt["top_context_columns"]
    sequences = chunk_indices(order_by_xyz(blocks), sequence_length)
    dataset = LowSequenceDataset(blocks, block_columns, target_columns, top_columns, sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_padded)
    pred_scaled = np.full((len(blocks), len(target_columns)), np.nan, dtype=np.float32)
    iterator = loader
    if tqdm is not None and show_progress:
        iterator = tqdm(loader, desc=f"{name} batches", leave=False)
    for batch in iterator:
        block = batch["block_features"].to(device)
        top = batch["top_context"].to(device)
        mask = batch["mask"].to(device)
        pred, _ = model.generate(
            block,
            top,
            mask=mask,
            decode_mode=decode_mode,
            temperature=softmax_temperature,
        )
        pred_np = pred.detach().cpu().numpy()
        order_np = batch["order"].numpy()
        mask_np = batch["mask"].numpy()
        for b in range(order_np.shape[0]):
            valid = mask_np[b]
            pred_scaled[order_np[b, valid]] = pred_np[b, valid]

    valid_rows = np.isfinite(pred_scaled).all(axis=1)
    pred_raw = target_scaler.inverse_array(pred_scaled[valid_rows])
    out = blocks.loc[valid_rows, ["X", "Y", "Z", "domain", "split"]].reset_index(drop=True)
    for idx, target in enumerate(target_scaler.columns):
        out[f"true_{target}"] = blocks.loc[valid_rows, target].to_numpy(dtype=float)
        out[f"pred_{target}"] = pred_raw[target].to_numpy(dtype=float)
        out[f"error_{target}"] = out[f"pred_{target}"] - out[f"true_{target}"]
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    model, ckpt = load_low_model(args.low_checkpoint, device)
    metadata = json.loads((args.prepared_dir / "metadata.json").read_text())
    target_scaler = TargetScaler.load(args.prepared_dir / "target_scaler.json")
    assays = pd.read_parquet(args.prepared_dir / "assays.parquet")

    top_model, top_features = load_top_model(Path(ckpt["top_checkpoint"]), device)
    top_prior_path = ckpt.get("top_prior_checkpoint", "")
    top_prior = None
    top_prior_ckpt = None
    assay_embeddings = None
    if top_prior_path:
        top_prior, top_prior_ckpt = load_top_prior(Path(top_prior_path), device)
    else:
        _, assay_embeddings = encode_assay_embeddings(assays, top_features, top_model, device)

    domains = []
    if args.domain in {"center_val", "both"}:
        center = pd.read_parquet(args.prepared_dir / "center_blocks.parquet")
        center = center.loc[center["split"] == "val"].reset_index(drop=True)
        if top_prior is not None and top_prior_ckpt is not None:
            center, _ = attach_prior_top_context(
                center,
                prior=top_prior,
                top_model=top_model,
                block_feature_columns=ckpt["block_feature_columns"],
                sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
                device=device,
            )
        else:
            center, _ = attach_top_context(center, assays, assay_embeddings, k=ckpt["model_config"]["top_k"])
        domains.append(("center_val", center))
    if args.domain in {"north_known", "both"}:
        north = pd.read_parquet(args.prepared_dir / "north_blocks.parquet")
        north = north.loc[north["split"] == "test_north_known"].reset_index(drop=True)
        if top_prior is not None and top_prior_ckpt is not None:
            north, _ = attach_prior_top_context(
                north,
                prior=top_prior,
                top_model=top_model,
                block_feature_columns=ckpt["block_feature_columns"],
                sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
                device=device,
            )
        else:
            north, _ = attach_top_context(north, assays, assay_embeddings, k=ckpt["model_config"]["top_k"])
        domains.append(("north_known", north))

    sequence_length = args.sequence_length or int(ckpt["model_config"]["sequence_length"])
    for name, blocks in domains:
        print(f"{name}: rows={len(blocks)}, sequence_length={sequence_length}")
        pred = predict_domain(
            name,
            blocks,
            model,
            ckpt,
            target_scaler,
            sequence_length,
            args.batch_size,
            device,
            not args.no_progress,
            args.decode_mode,
            args.softmax_temperature,
        )
        domain_dir = args.output_dir / name
        domain_dir.mkdir(parents=True, exist_ok=True)
        pred.to_csv(domain_dir / "predictions.csv", index=False)
        metrics = compute_metrics(pred, target_scaler.columns)
        metrics.to_csv(domain_dir / "metrics.csv", index=False)
        if not args.no_plots:
            plot_xy_maps(pred, metrics, target_scaler.columns, domain_dir, args.max_plot_points)
        print(metrics.to_string(index=False))
        print(f"Saved: {domain_dir}")


if __name__ == "__main__":
    main()

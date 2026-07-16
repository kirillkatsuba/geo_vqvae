from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .columns import TARGET_COLUMNS
from .evaluate import choose_device, load_low_model, predict_domain
from .preprocessing import TargetScaler
from .top_context import (
    attach_nearest_top_code,
    attach_prior_top_context,
    attach_prior_top_context_warm_start,
    attach_top_context,
    encode_assay_embeddings,
    load_top_model,
    load_top_prior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict target columns for all north block-model rows.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared_v2"))
    parser.add_argument("--low-checkpoint", type=Path, default=Path("geo_vqvae/runs/low_v7_soft_val/best_low.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("geo_vqvae/predictions/north_blocks_predicted.csv"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--decode-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--softmax-temperature", type=float, default=4.0)
    parser.add_argument("--warm-start-blocks", type=int, default=128)
    parser.add_argument(
        "--no-warm-start-center",
        action="store_true",
        help="Generate north independently instead of conditioning first north blocks on known center/south BM context.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def attach_context(
    blocks: pd.DataFrame,
    ckpt: dict,
    prepared_dir: Path,
    device: torch.device,
    warm_start_center: bool,
    warm_start_blocks: int,
) -> pd.DataFrame:
    assays = pd.read_parquet(prepared_dir / "assays.parquet")
    top_model, top_features = load_top_model(Path(ckpt["top_checkpoint"]), device)
    top_prior_path = ckpt.get("top_prior_checkpoint", "")
    if top_prior_path:
        top_prior, top_prior_ckpt = load_top_prior(Path(top_prior_path), device)
        if warm_start_center:
            center = pd.read_parquet(prepared_dir / "center_blocks.parquet")
            center = center.loc[center["has_targets"]].reset_index(drop=True)
            assay_codes, _ = encode_assay_embeddings(assays, top_features, top_model, device)
            center = attach_nearest_top_code(center, assays, assay_codes)
            blocks_with_context, _ = attach_prior_top_context_warm_start(
                blocks.reset_index(drop=True),
                context_blocks=center,
                context_codes=center["top_code_label"].to_numpy(),
                prior=top_prior,
                top_model=top_model,
                block_feature_columns=ckpt["block_feature_columns"],
                sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
                warm_start_length=warm_start_blocks,
                device=device,
            )
            return blocks_with_context
        blocks_with_context, _ = attach_prior_top_context(
            blocks.reset_index(drop=True),
            prior=top_prior,
            top_model=top_model,
            block_feature_columns=ckpt["block_feature_columns"],
            sequence_length=int(top_prior_ckpt["model_config"]["sequence_length"]),
            device=device,
        )
        return blocks_with_context

    _, assay_embeddings = encode_assay_embeddings(assays, top_features, top_model, device)
    blocks_with_context, _ = attach_top_context(
        blocks.reset_index(drop=True),
        assays,
        assay_embeddings,
        k=ckpt["model_config"]["top_k"],
    )
    return blocks_with_context


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model, ckpt = load_low_model(args.low_checkpoint, device)
    target_scaler = TargetScaler.load(args.prepared_dir / "target_scaler.json")

    north = pd.read_parquet(args.prepared_dir / "north_blocks.parquet").reset_index(drop=True)
    north = attach_context(
        north,
        ckpt,
        args.prepared_dir,
        device,
        warm_start_center=not args.no_warm_start_center,
        warm_start_blocks=args.warm_start_blocks,
    )

    sequence_length = args.sequence_length or int(ckpt["model_config"]["sequence_length"])
    print(
        "Predicting north blocks: "
        f"rows={len(north)}, sequence_length={sequence_length}, "
        f"decode_mode={args.decode_mode}, temperature={args.softmax_temperature}, "
        f"warm_start_center={not args.no_warm_start_center}, warm_start_blocks={args.warm_start_blocks}"
    )
    pred = predict_domain(
        "north_all",
        north,
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

    output = pred[["X", "Y", "Z"]].copy()
    for target in TARGET_COLUMNS:
        output[target] = pred[f"pred_{target}"].to_numpy()
    output = output[["X", "Y", "Z", "AS", "S", "CORG-1", "FE", "CA"]]
    output.to_csv(args.output_csv, index=False)
    print(f"Saved predictions: {args.output_csv}")
    print(f"Rows: {len(output)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a prepared dataset variant with local patch/neighborhood block features."
    )
    parser.add_argument("--prepared-dir", type=Path, default=Path("geo_vqvae/prepared_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/prepared_v2_patch"))
    parser.add_argument(
        "--neighbor-k",
        type=str,
        default="8,24",
        help="Comma-separated neighborhood sizes for local patch statistics.",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="mean,std,delta",
        help="Comma-separated stats: mean,std,delta. delta is current feature minus local mean.",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        default="",
        help="Optional comma-separated feature columns. Defaults to metadata block_feature_columns.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def parse_int_list(spec: str) -> list[int]:
    values = [int(item.strip()) for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one neighbor size")
    return sorted(set(values))


def parse_str_list(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def safe_stat_name(column: str) -> str:
    return (
        column.replace("feat_", "")
        .replace("=", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def neighbor_indices(df: pd.DataFrame, k: int) -> np.ndarray:
    coords = df[["X", "Y", "Z"]].to_numpy(dtype=np.float32)
    n_neighbors = min(k + 1, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    result = np.empty((len(df), min(k, max(1, len(df) - 1))), dtype=np.int64)
    for row_idx, row in enumerate(indices):
        without_self = row[row != row_idx]
        if len(without_self) == 0:
            without_self = row[:1]
        if len(without_self) < result.shape[1]:
            without_self = np.resize(without_self, result.shape[1])
        result[row_idx] = without_self[: result.shape[1]]
    return result


def add_patch_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    neighbor_sizes: list[int],
    stats: list[str],
    show_progress: bool,
) -> tuple[pd.DataFrame, list[str]]:
    values = df[feature_columns].to_numpy(dtype=np.float32)
    parts = []
    output_columns: list[str] = []
    iterator = neighbor_sizes
    if tqdm is not None and show_progress:
        iterator = tqdm(neighbor_sizes, desc="patch neighborhoods", leave=False)

    for k in iterator:
        idx = neighbor_indices(df, k)
        local = values[idx]
        local_mean = local.mean(axis=1)
        stat_frames = {}
        if "mean" in stats:
            stat_frames["mean"] = local_mean
        if "std" in stats:
            stat_frames["std"] = local.std(axis=1)
        if "delta" in stats:
            stat_frames["delta"] = values - local_mean

        for stat_name, stat_values in stat_frames.items():
            columns = [f"patch_k{k}_{stat_name}_{safe_stat_name(col)}" for col in feature_columns]
            output_columns.extend(columns)
            parts.append(pd.DataFrame(stat_values.astype(np.float32), columns=columns, index=df.index))

    if not parts:
        return df.copy(), []
    return pd.concat([df.copy(), *parts], axis=1), output_columns


def copy_side_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "assays.parquet",
        "assay_preprocessor.json",
        "block_preprocessor.json",
        "target_scaler.json",
        "summary.csv",
    ]:
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_side_files(args.prepared_dir, args.output_dir)

    metadata = json.loads((args.prepared_dir / "metadata.json").read_text())
    base_block_columns = list(metadata["block_feature_columns"])
    if args.feature_columns:
        feature_columns = parse_str_list(args.feature_columns)
    else:
        feature_columns = base_block_columns
    missing = [col for col in feature_columns if col not in base_block_columns]
    if missing:
        raise ValueError(f"Patch feature columns are not block features: {missing[:5]}")

    neighbor_sizes = parse_int_list(args.neighbor_k)
    stats = parse_str_list(args.stats)
    unsupported = sorted(set(stats) - {"mean", "std", "delta"})
    if unsupported:
        raise ValueError(f"Unsupported stats: {unsupported}")

    all_patch_columns: list[str] | None = None
    for filename in ["center_blocks.parquet", "north_blocks.parquet"]:
        frame = pd.read_parquet(args.prepared_dir / filename).reset_index(drop=True)
        print(
            f"Adding patch context to {filename}: rows={len(frame)}, "
            f"k={neighbor_sizes}, stats={stats}, features={len(feature_columns)}",
            flush=True,
        )
        augmented, patch_columns = add_patch_features(
            frame,
            feature_columns,
            neighbor_sizes,
            stats,
            show_progress=not args.no_progress,
        )
        if all_patch_columns is None:
            all_patch_columns = patch_columns
        elif all_patch_columns != patch_columns:
            raise RuntimeError("Patch columns differ between center and north")
        augmented.to_parquet(args.output_dir / filename, index=False)

    metadata["base_block_feature_columns"] = base_block_columns
    metadata["patch_context"] = {
        "neighbor_k": neighbor_sizes,
        "stats": stats,
        "source_feature_columns": feature_columns,
        "patch_feature_columns": all_patch_columns or [],
    }
    metadata["block_feature_columns"] = base_block_columns + (all_patch_columns or [])
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved patch-prepared data to: {args.output_dir}")
    print(f"Block features: {len(base_block_columns)} -> {len(metadata['block_feature_columns'])}")


if __name__ == "__main__":
    main()

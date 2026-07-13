from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import (
    AU_COL,
    BLOCK_CATEGORICAL_CANDIDATES,
    BLOCK_NUMERIC_CANDIDATES,
    CHEMICAL_CANDIDATES,
    COORD_COLUMNS,
    LITHOLOGY_CANDIDATES,
    TARGET_COLUMNS,
)
from .preprocessing import TargetScaler, TablePreprocessor, coerce_numeric, normalize_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare block-as-token VQ-VAE2 data.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("geo_vqvae/prepared"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--val-axis", choices=["X", "Y", "Z"], default="Y")
    parser.add_argument("--val-side", choices=["high", "low"], default="high")
    return parser.parse_args()


def read_assays(root: Path) -> pd.DataFrame:
    xlsx = root / "Вся_химия+литология+Au_final_all_data.XLSX"
    csv = root / "Вся_химия+литология+Au_final_all_data.csv"
    if xlsx.exists():
        df = pd.read_excel(xlsx)
    elif csv.exists():
        df = pd.read_csv(csv, low_memory=False)
    else:
        raise FileNotFoundError(f"Missing assays file under {root}")
    return normalize_columns(df)


def read_block(path: Path, domain: str) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path, low_memory=False))
    out = pd.DataFrame(index=df.index)
    for col in COORD_COLUMNS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["_X"] = pd.to_numeric(df.get("_X", df.get("_EAST", np.nan)), errors="coerce")
    out["_Y"] = pd.to_numeric(df.get("_Y", df.get("_NORTH", np.nan)), errors="coerce")
    out["_Z"] = pd.to_numeric(df.get("_Z", df.get("_RL", np.nan)), errors="coerce")
    out[AU_COL] = pd.to_numeric(df.get(AU_COL, np.nan), errors="coerce")
    for target in TARGET_COLUMNS:
        out[target] = pd.to_numeric(df.get(target, np.nan), errors="coerce")
    for col in ["DENSITY", "RESCAT", "MINED", "MODAREA", "ZONE", "PVALUE", "IND", "RESCAT_C"]:
        if col in df.columns:
            out[col] = df[col]
    out["domain"] = domain
    out["volume"] = out["_X"] * out["_Y"] * out["_Z"]
    out["has_targets"] = out[TARGET_COLUMNS].notna().all(axis=1)
    return out


def spatial_split(center: pd.DataFrame, val_fraction: float, axis: str, side: str) -> pd.Series:
    values = pd.to_numeric(center[axis], errors="coerce")
    q = 1.0 - val_fraction if side == "high" else val_fraction
    threshold = float(values.quantile(q))
    if side == "high":
        return values >= threshold
    return values <= threshold


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assays = read_assays(args.root)
    assay_numeric = sorted(set(CHEMICAL_CANDIDATES + COORD_COLUMNS + ["LITH1_VOL"]) & set(assays.columns))
    assays = coerce_numeric(assays, assay_numeric)

    assay_required = COORD_COLUMNS + [col for col in TARGET_COLUMNS if col in assays.columns]
    assays = assays.dropna(subset=COORD_COLUMNS).reset_index(drop=True)
    assays["has_targets"] = assays[[col for col in TARGET_COLUMNS if col in assays.columns]].notna().all(axis=1)

    center = read_block(args.root / "md_nat250721_CEN_Отработано.csv", domain="CEN")
    north = read_block(args.root / "md_nat241227(Модель_ресурсов_ind,inf).csv", domain="NTH")
    center = center.reset_index(drop=True)
    north = north.reset_index(drop=True)
    center["node_id"] = np.arange(len(center), dtype=np.int64)
    north["node_id"] = np.arange(len(north), dtype=np.int64)
    center_val_mask = spatial_split(center.loc[center["has_targets"]], args.val_fraction, args.val_axis, args.val_side)
    center["split"] = "unsupervised"
    known_idx = center.index[center["has_targets"]]
    center.loc[known_idx[center_val_mask.to_numpy()], "split"] = "val"
    center.loc[known_idx[~center_val_mask.to_numpy()], "split"] = "train"
    north["split"] = np.where(north["has_targets"], "test_north_known", "north_unknown")

    assay_numeric_cols = [c for c in CHEMICAL_CANDIDATES if c in assays.columns]
    assay_lith_cols = [c for c in LITHOLOGY_CANDIDATES if c in assays.columns and c not in assay_numeric_cols]
    assay_top_columns = []
    for col in assay_numeric_cols + COORD_COLUMNS + ["LITH1_VOL"]:
        if col in assays.columns and col not in assay_top_columns:
            assay_top_columns.append(col)
    assay_top_categorical = [col for col in assay_lith_cols if col in assays.columns]

    assay_pre = TablePreprocessor(assay_top_columns, assay_top_categorical).fit(assays)
    assay_features = assay_pre.transform(assays)

    block_numeric = [col for col in BLOCK_NUMERIC_CANDIDATES if col in center.columns or col in north.columns]
    block_categorical = [col for col in BLOCK_CATEGORICAL_CANDIDATES if col in center.columns or col in north.columns]
    block_schema_df = pd.concat([center, north], ignore_index=True)
    block_pre = TablePreprocessor(block_numeric, block_categorical).fit(block_schema_df)
    center_features = block_pre.transform(center)
    north_features = block_pre.transform(north)

    target_scaler = TargetScaler(TARGET_COLUMNS).fit(center.loc[center["split"] == "train"])
    center_scaled = target_scaler.transform(center)
    north_scaled = target_scaler.transform(north)
    center_scaled.columns = [f"{col}_scaled" for col in center_scaled.columns]
    north_scaled.columns = [f"{col}_scaled" for col in north_scaled.columns]

    assays_out = pd.concat([assays.reset_index(drop=True), assay_features.add_prefix("feat_")], axis=1)
    center_out = pd.concat([center, center_features.add_prefix("feat_"), center_scaled], axis=1)
    north_out = pd.concat([north, north_features.add_prefix("feat_"), north_scaled], axis=1)

    assays_out.to_parquet(args.output_dir / "assays.parquet", index=False)
    center_out.to_parquet(args.output_dir / "center_blocks.parquet", index=False)
    north_out.to_parquet(args.output_dir / "north_blocks.parquet", index=False)
    assay_pre.save(args.output_dir / "assay_preprocessor.json")
    block_pre.save(args.output_dir / "block_preprocessor.json")
    target_scaler.save(args.output_dir / "target_scaler.json")

    metadata = {
        "assay_feature_columns": [f"feat_{col}" for col in assay_pre.output_columns_],
        "block_feature_columns": [f"feat_{col}" for col in block_pre.output_columns_],
        "target_columns": TARGET_COLUMNS,
        "scaled_target_columns": [f"{col}_scaled" for col in TARGET_COLUMNS],
        "val_axis": args.val_axis,
        "val_side": args.val_side,
        "val_fraction": args.val_fraction,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    summary = pd.DataFrame(
        [
            {"dataset": "assays", "rows": len(assays_out), "known_targets": int(assays_out["has_targets"].sum())},
            {"dataset": "center_train", "rows": int((center_out["split"] == "train").sum()), "known_targets": int((center_out["split"] == "train").sum())},
            {"dataset": "center_val", "rows": int((center_out["split"] == "val").sum()), "known_targets": int((center_out["split"] == "val").sum())},
            {"dataset": "north_known", "rows": int((north_out["split"] == "test_north_known").sum()), "known_targets": int((north_out["split"] == "test_north_known").sum())},
            {"dataset": "north_unknown", "rows": int((north_out["split"] == "north_unknown").sum()), "known_targets": 0},
        ]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Assay top features: {len(metadata['assay_feature_columns'])}")
    print(f"Block features: {len(metadata['block_feature_columns'])}")
    print(f"Saved prepared data to: {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    from .columns import normalize_column_map

    return df.rename(columns=normalize_column_map())


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            if out[col].dtype == "object":
                out[col] = out[col].astype(str).str.replace(",", ".", regex=False)
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class TablePreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str] = field(default_factory=list)
    include_missing_indicators: bool = True
    medians_: dict[str, float] = field(default_factory=dict)
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "TablePreprocessor":
        for col in self.numeric_columns:
            values = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians_[col] = median
            self.means_[col] = mean
            self.stds_[col] = std if np.isfinite(std) and std > 1e-12 else 1.0

        for col in self.categorical_columns:
            values = df.get(col, pd.Series(index=df.index, dtype=object)).astype("string").fillna("__MISSING__")
            self.categories_[col] = sorted(values.unique().tolist())

        self.output_columns_ = self.transform(df).columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col in self.numeric_columns:
            values = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce")
            missing = values.isna().astype(float)
            filled = values.fillna(self.medians_.get(col, 0.0))
            standardized = (filled - self.means_.get(col, 0.0)) / self.stds_.get(col, 1.0)
            parts.append(pd.DataFrame({col: standardized}, index=df.index))
            if self.include_missing_indicators:
                parts.append(pd.DataFrame({f"{col}__missing": missing}, index=df.index))

        for col, categories in self.categories_.items():
            values = df.get(col, pd.Series(index=df.index, dtype=object)).astype("string").fillna("__MISSING__")
            encoded = {f"{col}={category}": (values == category).astype(float) for category in categories}
            parts.append(pd.DataFrame(encoded, index=df.index))

        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
        if self.output_columns_:
            for col in self.output_columns_:
                if col not in out.columns:
                    out[col] = 0.0
            out = out[self.output_columns_]
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "include_missing_indicators": self.include_missing_indicators,
            "medians": self.medians_,
            "means": self.means_,
            "stds": self.stds_,
            "categories": self.categories_,
            "output_columns": self.output_columns_,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TablePreprocessor":
        obj = cls(
            numeric_columns=list(data["numeric_columns"]),
            categorical_columns=list(data.get("categorical_columns", [])),
            include_missing_indicators=bool(data.get("include_missing_indicators", True)),
        )
        obj.medians_ = dict(data.get("medians", {}))
        obj.means_ = dict(data.get("means", {}))
        obj.stds_ = dict(data.get("stds", {}))
        obj.categories_ = {key: list(value) for key, value in data.get("categories", {}).items()}
        obj.output_columns_ = list(data.get("output_columns", []))
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TablePreprocessor":
        return cls.from_dict(json.loads(path.read_text()))


@dataclass
class TargetScaler:
    columns: list[str]
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "TargetScaler":
        for col in self.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            self.mean_[col] = mean
            self.std_[col] = std if np.isfinite(std) and std > 1e-12 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: (pd.to_numeric(df[col], errors="coerce") - self.mean_[col]) / self.std_[col]
                for col in self.columns
            },
            index=df.index,
            dtype=np.float32,
        )

    def inverse_array(self, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: values[:, idx] * self.std_[col] + self.mean_[col]
                for idx, col in enumerate(self.columns)
            }
        )

    def to_dict(self) -> dict:
        return {"columns": self.columns, "mean": self.mean_, "std": self.std_}

    @classmethod
    def from_dict(cls, data: dict) -> "TargetScaler":
        obj = cls(columns=list(data["columns"]))
        obj.mean_ = dict(data["mean"])
        obj.std_ = dict(data["std"])
        return obj

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TargetScaler":
        return cls.from_dict(json.loads(path.read_text()))

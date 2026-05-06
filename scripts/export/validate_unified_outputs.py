#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate unified CSV exports against the frozen schema manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "unified_output_schema_manifest.json"


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns: {missing}")


def require_constant(df: pd.DataFrame, column: str, expected: str, table_name: str) -> None:
    values = sorted({str(v) for v in df[column].dropna().unique()})
    if values != [expected]:
        raise ValueError(f"{table_name}: column {column} expected only {expected}, got {values}")


def validate_per_row(df: pd.DataFrame, table_name: str) -> None:
    if df["uid"].duplicated().any():
        dupes = df.loc[df["uid"].duplicated(), "uid"].head(10).tolist()
        raise ValueError(f"{table_name}: duplicated uid values: {dupes}")


def validate_group_level(df: pd.DataFrame, table_name: str) -> None:
    keys = ["run_id", "model_id", "task", "env_task", "group_id"]
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{table_name}: duplicated group-level keys")


def validate_model_level(df: pd.DataFrame, table_name: str) -> None:
    keys = ["run_id", "model_id", "slice_kind", "task", "env_task"]
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{table_name}: duplicated model-level keys")


def validate_figure_ready(df: pd.DataFrame, table_name: str) -> None:
    if df["plot_id"].isna().any() or (df["plot_id"].astype(str).str.strip() == "").any():
        raise ValueError(f"{table_name}: empty plot_id values found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unified CSV exports")
    parser.add_argument("--export-dir", required=True, help="Directory containing unified CSV exports")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Schema manifest JSON path")
    args = parser.parse_args()

    export_dir = Path(args.export_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    schema_version = manifest["schema_version"]

    validators = {
        "per-row.csv": validate_per_row,
        "group-level.csv": validate_group_level,
        "model-level.csv": validate_model_level,
        "figure-ready.csv": validate_figure_ready,
    }

    for table_name, spec in manifest["tables"].items():
        csv_path = export_dir / table_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing export table: {csv_path}")
        df = pd.read_csv(csv_path)
        require_columns(df, spec["required_columns"], table_name)
        require_constant(df, "output_schema_version", schema_version, table_name)
        require_constant(df, "aggregation_level", spec["aggregation_level"], table_name)
        validators[table_name](df, table_name)

    print(f"Unified output validation passed for {export_dir}")


if __name__ == "__main__":
    main()

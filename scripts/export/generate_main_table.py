#!/usr/bin/env python3
"""Generate SSOT main table from legacy inputs or unified CSV exports."""

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def load_jsonl(path: Path) -> List[dict]:
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV file"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _metric_key(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_env_from_exam_id(exam_id: str) -> str:
    parts = str(exam_id).split(".")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def _build_jaccneu_from_responses(responses_path: Path) -> Dict[str, float]:
    jaccneu_by_env = {}
    for resp in load_jsonl(responses_path):
        exam_id = resp.get("exam_id", "")
        env = extract_env_from_exam_id(exam_id)
        response_text = str(resp.get("pred", resp.get("raw", "")))

        if "success" in response_text.lower() and "fail" not in response_text.lower():
            pred = 1
        elif "fail" in response_text.lower():
            pred = 0
        else:
            pred = -1

        exam_meta = resp.get("meta", {}).get("exam", {})
        label = exam_meta.get("label", -1)
        if pred >= 0 and label >= 0:
            jaccneu_by_env.setdefault(env, {"correct": 0, "total": 0})
            jaccneu_by_env[env]["total"] += 1
            if pred == label:
                jaccneu_by_env[env]["correct"] += 1

    return {
        env: (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
        for env, stats in jaccneu_by_env.items()
    }


def _build_jaccneu_from_model_level(model_level_csv: Path) -> Dict[str, float]:
    """
    Read JAccneu (correctness anchor) from unified model-level CSV.

    JAccneu is computed from task == "D" (full.orig.clean correctness rate).
    Legacy Task C artifact also has full.orig.clean but the primary measurement
    task for JAccneu is Task D (SSOT §10.0).
    """
    model_df = load_csv(model_level_csv)
    if model_df.empty:
        return {}

    # Primary: JAccneu from Task D
    d_subset = model_df[
        (model_df["aggregation_level"] == "model_level")
        & (model_df["slice_kind"] == "task_env")
        & (model_df["task"] == "D")
    ].copy()
    jaccneu = {
        _metric_key(row["env_task"]): float(row["acc"])
        for _, row in d_subset.iterrows()
        if _metric_key(row["env_task"])
    }
    # Fallback: JAccneu from legacy Task C full.orig.clean (if Task D not yet run)
    if not jaccneu:
        c_subset = model_df[
            (model_df["aggregation_level"] == "model_level")
            & (model_df["slice_kind"] == "task_env")
            & (model_df["task"] == "C")
        ].copy()
        jaccneu = {
            _metric_key(row["env_task"]): float(row["acc"])
            for _, row in c_subset.iterrows()
            if _metric_key(row["env_task"])
        }
    return jaccneu


def _build_main_table_rows(
    jaccneu_by_env: Dict[str, float],
    idr_csv: Path,
    les_csv: Path,
    quadrant_csv: Path,
    vcc_csv: Path = None,
    pixel_csv: Path = None,
    model_name: str = None
) -> pd.DataFrame:
    rows = []

    # Load IDR metrics
    idr_df = load_csv(idr_csv)
    idr_by_env = {}
    if not idr_df.empty:
        for _, row in idr_df.iterrows():
            env = _metric_key(row.get('env', row.get('task', 'unknown')))
            idr_by_env[env] = {
                'IDR_raw': row.get('IDR_raw', row.get('idr_raw', 0)),
                'IDR_cond': row.get('IDR_cond', row.get('idr_cond', 0)),
                'G_active': row.get('G_active_pct', row.get('g_active', 0)),
                'Degenerate': row.get('Degenerate', row.get('degenerate', False))
            }

    # Load LES metrics
    les_df = load_csv(les_csv)
    les_by_env = {}
    if not les_df.empty:
        for _, row in les_df.iterrows():
            task = _metric_key(row.get('task', 'unknown'))
            family = row.get('family', '')
            if task not in les_by_env:
                les_by_env[task] = {}
            les_by_env[task][family] = row.get('LES_beta', row.get('les_beta', 0))

    # Load NS metrics from quadrant
    quadrant_df = load_csv(quadrant_csv)
    ns_by_env = {}
    if not quadrant_df.empty:
        for _, row in quadrant_df.iterrows():
            env = _metric_key(row.get('env', row.get('task', 'unknown')))
            ns_by_env[env] = {
                'NS_T': row.get('NS_T', row.get('ns_t', 0)),
                'NS_vis': row.get('NS_vis', row.get('ns_vis', 0)),
                'NS_frame': row.get('NS_frame', row.get('ns_frame', 0))
            }

    # Load VCC oracle
    vcc_by_env = {}
    if vcc_csv and vcc_csv.exists():
        vcc_df = load_csv(vcc_csv)
        if not vcc_df.empty:
            for _, row in vcc_df.iterrows():
                vcc_by_env['overall'] = row.get('vcc', 0)

    # Load pixel baseline
    pixel_by_env = {}
    if pixel_csv and pixel_csv.exists():
        pixel_df = load_csv(pixel_csv)
        if not pixel_df.empty:
            for _, row in pixel_df.iterrows():
                task = _metric_key(row.get('task', 'unknown'))
                pixel_by_env[task] = {
                    'pixel_baseline_acc': row.get('acc_ssim', 0),
                    'pixel_gate_passed': row.get('gate_passed', False)
                }

    # Get all environments
    all_envs = (
        set(jaccneu_by_env.keys())
        | set(idr_by_env.keys())
        | set(les_by_env.keys())
        | set(ns_by_env.keys())
        | set(pixel_by_env.keys())
    )

    # Build final table
    for env in sorted(all_envs):
        row = {
            'model': model_name or 'unknown',
            'task_env': env,
            'JAccneu': jaccneu_by_env.get(env, 0),
            'IDR_raw': idr_by_env.get(env, {}).get('IDR_raw', 0),
            'IDR_cond': idr_by_env.get(env, {}).get('IDR_cond', 0),
            'G_active_pct': idr_by_env.get(env, {}).get('G_active', 0),
            'NS_T': ns_by_env.get(env, {}).get('NS_T', 0),
            'NS_vis': ns_by_env.get(env, {}).get('NS_vis', 0),
            'NS_frame': ns_by_env.get(env, {}).get('NS_frame', 0),
            'LES_T': les_by_env.get(env, {}).get('temporal', 0),
            'LES_vis': les_by_env.get(env, {}).get('visual', 0),
            'LES_frame': les_by_env.get(env, {}).get('framing', 0),
            'Degenerate': idr_by_env.get(env, {}).get('Degenerate', False),
            'VCC_oracle': vcc_by_env.get('overall', None),
            'pixel_baseline_acc': pixel_by_env.get(env, {}).get('pixel_baseline_acc', None),
            'pixel_gate_passed': pixel_by_env.get(env, {}).get('pixel_gate_passed', None)
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_main_table_legacy(
    responses_path: Path,
    idr_csv: Path,
    les_csv: Path,
    quadrant_csv: Path,
    vcc_csv: Path = None,
    pixel_csv: Path = None,
    model_name: str = None
) -> pd.DataFrame:
    return _build_main_table_rows(
        jaccneu_by_env=_build_jaccneu_from_responses(responses_path),
        idr_csv=idr_csv,
        les_csv=les_csv,
        quadrant_csv=quadrant_csv,
        vcc_csv=vcc_csv,
        pixel_csv=pixel_csv,
        model_name=model_name,
    )


def generate_main_table_from_unified(
    model_level_csv: Path,
    idr_csv: Path,
    les_csv: Path,
    quadrant_csv: Path,
    vcc_csv: Path = None,
    pixel_csv: Path = None,
    model_name: str = None
) -> pd.DataFrame:
    model_df = load_csv(model_level_csv)
    inferred_model = model_name
    if not inferred_model and not model_df.empty and "model_id" in model_df.columns:
        inferred_model = _metric_key(model_df["model_id"].iloc[0]) or None

    return _build_main_table_rows(
        jaccneu_by_env=_build_jaccneu_from_model_level(model_level_csv),
        idr_csv=idr_csv,
        les_csv=les_csv,
        quadrant_csv=quadrant_csv,
        vcc_csv=vcc_csv,
        pixel_csv=pixel_csv,
        model_name=inferred_model,
    )


def check_data_quality(df: pd.DataFrame) -> List[str]:
    """
    Run data quality checks and return warnings.
    """
    warnings = []

    for _, row in df.iterrows():
        env = row.get('task_env', row.get('env', 'unknown'))

        # Check 1: JAccneu should be > random (0.5 for binary)
        jac = row.get('JAccneu', 0)
        if jac < 0.4:
            warnings.append(f"[CRITICAL] {env}: JAccneu={jac:.3f} < 0.4 - accuracy too low, check prompt/data")

        # Check 2: IDR_cond should be > 0 for valid intervention
        idr_cond = row.get('IDR_cond', 0)
        if idr_cond <= 0:
            warnings.append(f"[WARNING] {env}: IDR_cond={idr_cond:.3f} <= 0 - intervention not effective")

        # Check 3: Pixel baseline should fail (acc < 0.35)
        pixel_acc = row.get('pixel_baseline_acc')
        if pixel_acc is not None and not pd.isna(pixel_acc):
            if pixel_acc > 0.35:
                warnings.append(f"[CRITICAL] {env}: pixel_baseline_acc={pixel_acc:.3f} > 0.35 - "
                              f"Task A is TRIVIALLY SEPARABLE, negative samples are too easy!")

        # Check 4: LES should be positive for valid leakage
        les_t = row.get('LES_T', 0)
        if les_t < 0:
            warnings.append(f"[WARNING] {env}: LES_T={les_t:.3f} < 0 - temporal leakage negative?")

        # Check 5: Degenerate flag
        degenerate = row.get('Degenerate', False)
        if degenerate not in (False, None, "", "False") and not pd.isna(degenerate):
            warnings.append(f"[INFO] {env}: Degenerate={degenerate} - interpret intervention stats with caution")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="Generate Main Results Table")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--responses", "-r", help="Responses JSONL path")
    source_group.add_argument("--model-level", help="Unified model-level CSV path")
    parser.add_argument("--idr", "-i", required=True, help="IDR CSV path")
    parser.add_argument("--les", "-l", required=True, help="LES CSV path")
    parser.add_argument("--quadrant", "-q", required=True, help="NS-IDR quadrant CSV path")
    parser.add_argument("--vcc", help="VCC oracle CSV path (optional)")
    parser.add_argument("--pixel", help="Pixel baseline CSV path (optional)")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    args = parser.parse_args()

    # Generate table
    if args.model_level:
        df = generate_main_table_from_unified(
            model_level_csv=Path(args.model_level),
            idr_csv=Path(args.idr),
            les_csv=Path(args.les),
            quadrant_csv=Path(args.quadrant),
            vcc_csv=Path(args.vcc) if args.vcc else None,
            pixel_csv=Path(args.pixel) if args.pixel else None,
            model_name=args.model,
        )
    else:
        df = generate_main_table_legacy(
            responses_path=Path(args.responses),
            idr_csv=Path(args.idr),
            les_csv=Path(args.les),
            quadrant_csv=Path(args.quadrant),
            vcc_csv=Path(args.vcc) if args.vcc else None,
            pixel_csv=Path(args.pixel) if args.pixel else None,
            model_name=args.model,
        )

    # Run quality checks
    warnings = check_data_quality(df)

    # Print table
    print("\n" + "=" * 100)
    print("Main Results Table (SSOT Table 1)")
    print("=" * 100)
    print(df.to_string(index=False))

    # Print warnings
    if warnings:
        print("\n" + "=" * 100)
        print("DATA QUALITY WARNINGS")
        print("=" * 100)
        for w in warnings:
            print(w)

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

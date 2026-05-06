#!/usr/bin/env python3
"""
IDR (Interventional Discrimination Rate) Calculator

Reference: Stats Spec §4
Computes C4 core metrics: IDR_raw, IDR_cond, G_active%, Degenerate detection
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent
# Auto-add repo root so "scripts.xxx" imports work from any subdirectory
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.schema.exam_schema import parse_verdict as _parse_verdict_ssot


@dataclass
class IDRResult:
    idr_raw: float
    idr_cond: float
    g_active: int
    g_total: int
    active_rate: float
    idr_rev: float  # reverse IDR
    degenerate: Optional[str]  # "all_fail" / "all_success" / None
    method: str = "standard"


def _is_blank_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip() == ""


def _baseline_condition_mask(df: pd.DataFrame, variant: str) -> pd.Series:
    """
    Select rows for a given variant.

    Legacy Task C (has temporal/visual): additionally filter to baseline nuisance
    condition: `temporal=orig`, `visual=clean`, no framing probe.

    Task D (no temporal/visual): variant-only filtering is sufficient — Task D
    has no temporal/visual/framing probe dimensions by design.
    """
    if "temporal" in df.columns and df["temporal"].notna().any():
        # Legacy Task C: filter by temporal + visual + framing
        temporal_ok = df["temporal"].apply(lambda v: _is_blank_like(v) or str(v).strip() == "orig")
        visual_ok = df["visual"].apply(lambda v: _is_blank_like(v) or str(v).strip() == "clean")
        framing_ok = df["framing"].apply(_is_blank_like) if "framing" in df.columns else True
        return (df["variant"] == variant) & temporal_ok & visual_ok & framing_ok
    else:
        # Task D: variant-only (no temporal/visual columns)
        return df["variant"] == variant


def _dedupe_group_rows(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    subset = df[_baseline_condition_mask(df, variant)].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(["group_id", "exam_id"]).drop_duplicates(subset=["group_id"], keep="first")
    return subset


def load_verdicts(responses_path: str, exam_path: str = None) -> pd.DataFrame:
    """
    Load verdict data and merge with exam.

    Args:
        responses_path: Inference results JSONL path
        exam_path: Exam JSONL path (optional, for getting ground truth)

    Returns:
        DataFrame with columns: exam_id, model, variant, group_id, y_pred, y_true, framing
        framing: parsed from exam_id 7th part (.pos/.neu/.neg) or '' if not present
    """
    verdicts = []

    with open(responses_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)

            exam_id = record.get('exam_id', record.get('uid', ''))
            # Parse exam_id: C.<task>.<group_id>.<variant>.<temporal>.<visual>[.<framing>]
            # add_framing_probe.py adds 7th part: .pos/.neu/.neg
            parts = exam_id.split('.')
            if len(parts) >= 3:
                task = parts[1] if len(parts) > 1 else ''
                group_id = parts[2] if len(parts) > 2 else ''
                variant = parts[3] if len(parts) > 3 else ''
                temporal = parts[4] if len(parts) > 4 else ''
                visual = parts[5] if len(parts) > 5 else ''
                # P0-4: framing parsing (added by add_framing_probe.py as .pos/.neu/.neg)
                framing = parts[6] if len(parts) > 6 else ''
            else:
                task = group_id = variant = temporal = visual = framing = ''

            # Parse model response - prefer pred field, fall back to raw
            response = record.get('pred', record.get('raw', record.get('response', '')))
            pred_success = parse_verdict(response)

            # Model name may be in meta.model or top-level model field
            model_name = record.get('model', record.get('meta', {}).get('model', 'unknown'))

            verdicts.append({
                'exam_id': exam_id,
                'model': model_name,
                'task': task,
                'group_id': group_id,
                'variant': variant,
                'temporal': temporal,
                'visual': visual,
                'framing': framing,  # P0-4: new, parsed from UID 7th part
                'y_pred': pred_success,  # 1 = Success, 0 = Fail
                'response': response
            })

    return pd.DataFrame(verdicts)


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _coerce_pred_label(pred_norm: Any, fallback_text: Any) -> int:
    pred_text = _clean_text(pred_norm)
    if pred_text == "Success":
        return 1
    if pred_text == "Fail":
        return 0
    return parse_verdict(_clean_text(fallback_text))


def load_verdicts_from_per_row(per_row_csv: str) -> pd.DataFrame:
    """
    Load verdicts from unified `per-row.csv`.

    Keeps the legacy output contract of `load_verdicts()` so downstream IDR/NS
    calculations remain unchanged while the preferred input shifts to unified CSV.
    """
    df = pd.read_csv(per_row_csv)
    if df.empty:
        return pd.DataFrame(columns=[
            "exam_id", "model", "task", "group_id", "variant",
            "temporal", "visual", "framing", "y_pred", "response",
        ])

    if "task" in df.columns:
        # IDR is computable on both legacy Task C and Task D (both have cf variant).
        # Legacy C: temporal/visual probe richness; Task D: continuous outcome judgment.
        # Filter to both so callers can compute IDR for whichever artifact they have.
        df = df[df["task"].isin(["C", "D"])].copy()

    verdicts = []
    for _, row in df.iterrows():
        response = _clean_text(row.get("response_text_pred")) or _clean_text(row.get("response_text_raw"))
        verdicts.append({
            "exam_id": _clean_text(row.get("exam_id")) or _clean_text(row.get("uid")),
            "model": _clean_text(row.get("model_id")) or "unknown",
            # BUG FIX (Phase 32): was row.get("env_task") — incorrectly used env as task label.
            # This caused ALL Task D rows to be mislabeled as task="doorkey/keycorridor/etc"
            # and filtered out by the task in ["C","D"] filter above.
            # Correct: use the task field from the per-row CSV (written by score_exam.py).
            "task": _clean_text(row.get("task")) or "unknown",
            "group_id": _clean_text(row.get("group_id")),
            "variant": _clean_text(row.get("variant")),
            "temporal": _clean_text(row.get("temporal")),
            "visual": _clean_text(row.get("visual")),
            "framing": _clean_text(row.get("framing")),
            "y_pred": _coerce_pred_label(row.get("pred_norm"), response),
            "response": response,
        })

    return pd.DataFrame(verdicts)


def parse_verdict(response: str) -> int:
    """P0-2 fix: delegate to shared exam_schema helper (matches score_exam.py)."""
    ans, mode = _parse_verdict_ssot(response)
    if ans == "Success":
        return 1
    if ans == "Fail":
        return 0
    return -1


def compute_idr(verdicts_df: pd.DataFrame, model: str = None) -> Dict[str, IDRResult]:
    """
    Compute IDR metrics.
    
    Args:
        verdicts_df: DataFrame containing variant, group_id, y_pred
        model: Optional, filter by model
    
    Returns:
        IDR results dictionary grouped by model
    """
    if model:
        verdicts_df = verdicts_df[verdicts_df['model'] == model].copy()
    
    results = {}
    
    # Group by model
    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}
        
        # Group by task
        for task, task_df in model_df.groupby('task'):
            # IDR only uses the baseline nuisance condition:
            # full.orig.clean.(no framing) vs cf.orig.clean.(no framing)
            full_df = _dedupe_group_rows(task_df, 'full').set_index('group_id')
            cf_df = _dedupe_group_rows(task_df, 'cf').set_index('group_id')
            
            # Find common groups
            common_groups = full_df.index.intersection(cf_df.index)
            
            if len(common_groups) == 0:
                continue
            
            g_total = len(common_groups)
            
            # Compute G_active (groups where model predicts Success on Full)
            full_success = full_df.loc[common_groups, 'y_pred'] == 1
            g_active = int(full_success.sum())
            active_rate = g_active / g_total if g_total > 0 else 0.0
            
            # IDR_raw = P(ŷ_full=Success ∧ ŷ_cf=Fail)
            cf_fail = cf_df.loc[common_groups, 'y_pred'] == 0
            idr_raw = ((full_success) & (cf_fail)).sum() / g_total
            
            # IDR_cond = IDR_raw × G / G_active (computed only on groups where Full=Success)
            if g_active > 0:
                idr_cond = idr_raw * g_total / g_active
            else:
                idr_cond = float('nan')
            
            # IDR_rev = P(ŷ_full=Fail ∧ ŷ_cf=Success)
            full_fail = full_df.loc[common_groups, 'y_pred'] == 0
            cf_success = cf_df.loc[common_groups, 'y_pred'] == 1
            idr_rev = ((full_fail) & (cf_success)).sum() / g_total
            
            # Degenerate detection
            degenerate = None
            if active_rate < 0.1:
                degenerate = "all_fail"
            elif active_rate > 0.9:
                cf_succ_rate = cf_df.loc[common_groups, 'y_pred'].mean()
                if cf_succ_rate > 0.9:
                    degenerate = "all_success"
            
            model_results[task] = IDRResult(
                idr_raw=idr_raw,
                idr_cond=idr_cond,
                g_active=g_active,
                g_total=g_total,
                active_rate=active_rate,
                idr_rev=idr_rev,
                degenerate=degenerate
            )
        
        results[model_name] = model_results
    
    return results


def compute_ns(verdicts_df: pd.DataFrame, baseline_variant: str = 'full') -> Dict[str, Dict[str, float]]:
    """
    Compute Nuisance Stability (NS).

    NS_F = P(y_full_baseline = y_probe) grouped by family.

    P0-4 fix: Added framing column filtering.
    - add_framing_probe.py generates framing variants (.pos/.neu/.neg) that have
      temporal=orig, visual=clean but different framing. These are FRAMING family probes,
      NOT Visual family probes. Including them in NS_vis corrupts the visual family metric.
    - Temporal family: temporal != orig (rev, blockswap_*), framing is ''
    - Visual family: visual != clean (noisy, style_*), framing is ''
    - Framing family: framing in (pos, neu, neg), temporal=orig, visual=clean

    Args:
        verdicts_df: DataFrame containing variant, group_id, temporal, visual, framing, y_pred
        baseline_variant: Baseline variant name

    Returns:
        NS dictionary grouped by model/task/family
    """
    results = {}

    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}

        for task, task_df in model_df.groupby('task'):
            task_results = {}

            # Get baseline verdicts (variant=full and temporal=orig, visual=clean)
            baseline_df = _dedupe_group_rows(task_df, baseline_variant)
            baseline_by_group = dict(zip(baseline_df['group_id'], baseline_df['y_pred']))

            if not baseline_by_group:
                model_results[task] = task_results
                continue

            # Collect all probe NS
            probe_types = []

            # Temporal probes: temporal != orig (not baseline), variant=full
            # framing should be empty for temporal probes
            temporal_probes = task_df[
                task_df['temporal'].notna() &
                (task_df['temporal'] != '') &
                (task_df['temporal'] != 'orig') &
                (task_df['variant'] == baseline_variant) &
                (task_df['framing'].isin(['', None]))  # framing probes are a separate family
            ]
            for temp_probe in temporal_probes['temporal'].unique():
                probe_df = temporal_probes[temporal_probes['temporal'] == temp_probe]
                matches = 0
                total = 0
                for _, row in probe_df.iterrows():
                    gid = row['group_id']
                    if gid in baseline_by_group:
                        if row['y_pred'] == baseline_by_group[gid]:
                            matches += 1
                        total += 1
                if total > 0:
                    task_results[f'NS_{temp_probe}'] = matches / total
                probe_types.append(('temporal', temp_probe))

            # Visual probes: visual != clean (not baseline), framing is empty (P0-4)
            # CRITICAL: exclude framing variants (.pos/.neu/.neg) from visual family
            # These have visual=clean but framing=pos/neu/neg → Framing family, not Visual
            visual_probes = task_df[
                task_df['visual'].notna() &
                (task_df['visual'] != '') &
                (task_df['visual'] != 'clean') &
                (task_df['variant'] == baseline_variant) &
                (task_df['framing'].isin(['', None]))  # P0-4: exclude framing variants
            ]
            for vis_probe in visual_probes['visual'].unique():
                probe_df = visual_probes[visual_probes['visual'] == vis_probe]
                matches = 0
                total = 0
                for _, row in probe_df.iterrows():
                    gid = row['group_id']
                    if gid in baseline_by_group:
                        if row['y_pred'] == baseline_by_group[gid]:
                            matches += 1
                        total += 1
                if total > 0:
                    task_results[f'NS_{vis_probe}'] = matches / total
                probe_types.append(('visual', vis_probe))

            # P0-4: Framing family NS (new — was previously missing entirely)
            # Framing family: framing in (pos, neu, neg), temporal=orig, visual=clean
            framing_probes = task_df[
                task_df['framing'].isin(['pos', 'neu', 'neg']) &
                (task_df['variant'] == baseline_variant) &
                (task_df['temporal'].isin(['', 'orig'])) &
                (task_df['visual'].isin(['', 'clean']))
            ]
            for frame_probe in framing_probes['framing'].unique():
                probe_df = framing_probes[framing_probes['framing'] == frame_probe]
                matches = 0
                total = 0
                for _, row in probe_df.iterrows():
                    gid = row['group_id']
                    if gid in baseline_by_group:
                        if row['y_pred'] == baseline_by_group[gid]:
                            matches += 1
                        total += 1
                if total > 0:
                    task_results[f'NS_{frame_probe}'] = matches / total

            model_results[task] = task_results

        results[model_name] = model_results

    return results


def export_quadrant_data_from_df(verdicts_df: pd.DataFrame, output_path: str):
    """
    Export data required for NS-IDR quadrant plot.
    
    Output CSV contains: model, task, NS, IDR_raw, IDR_cond, G_active%, Degenerate
    """
    idr_results = compute_idr(verdicts_df)
    ns_results = compute_ns(verdicts_df)
    
    rows = []
    for model, model_idr in idr_results.items():
        model_ns = ns_results.get(model, {})
        for task, idr in model_idr.items():
            ns = model_ns.get(task, {})
            # Average NS
            avg_ns = np.mean([v for k, v in ns.items()]) if ns else float('nan')
            
            rows.append({
                'model': model,
                'task': task,
                'NS': avg_ns,
                'IDR_raw': idr.idr_raw,
                'IDR_cond': idr.idr_cond if not np.isnan(idr.idr_cond) else None,
                'G_active_pct': idr.active_rate,
                'Degenerate': idr.degenerate
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported quadrant data to {output_path}")
    return df


def export_quadrant_data(responses_path: str, output_path: str):
    verdicts_df = load_verdicts(responses_path)
    return export_quadrant_data_from_df(verdicts_df, output_path)


def print_idr_results(results: Dict[str, Dict[str, IDRResult]]):
    """Print IDR results"""
    for model, model_results in results.items():
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        print(f"{'Task':<15} {'IDR_raw':>10} {'IDR_cond':>10} {'G_active%':>12} {'Degenerate':>12}")
        print("-" * 60)
        for task, r in model_results.items():
            deg = r.degenerate if r.degenerate else '-'
            idr_c = f"{r.idr_cond:.3f}" if not np.isnan(r.idr_cond) else "N/A"
            print(f"{task:<15} {r.idr_raw:>10.3f} {idr_c:>10} {r.active_rate:>11.1%} {deg:>12}")


def main():
    parser = argparse.ArgumentParser(description="IDR Calculator")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--responses", "-r", help="Responses JSONL path")
    source_group.add_argument("--per-row-csv", help="Unified per-row CSV path")
    parser.add_argument("--model", "-m", help="Filter by model name")
    parser.add_argument("--output-csv", "-o", help="Output CSV path")
    args = parser.parse_args()

    if args.per_row_csv:
        verdicts_df = load_verdicts_from_per_row(args.per_row_csv)
    else:
        verdicts_df = load_verdicts(args.responses)
    results = compute_idr(verdicts_df, args.model)
    
    print_idr_results(results)
    
    if args.output_csv:
        export_quadrant_data_from_df(verdicts_df, args.output_csv)


if __name__ == "__main__":
    main()

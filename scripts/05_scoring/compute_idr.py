#!/usr/bin/env python3
"""
IDR (Interventional Discrimination Rate) Calculator

参考 Stats Spec §4
计算 C4 核心指标：IDR_raw, IDR_cond, G_active%, Degenerate 检测
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
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.exam_schema import parse_verdict as _parse_verdict_ssot


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
        framing_ok = (
            df["framing"].apply(
                lambda v: _is_blank_like(v) or str(v).strip() == "neu"
            )
            if "framing" in df.columns
            else True
        )
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
    加载 verdict 数据并与 exam 合并

    Args:
        responses_path: 推理结果 JSONL 路径
        exam_path: exam JSONL 路径（可选，用于获取 ground truth）

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
            # 解析 exam_id: C.<task>.<group_id>.<variant>.<temporal>.<visual>[.<framing>]
            # add_framing_probe.py adds 7th part: .pos/.neu/.neg
            parts = exam_id.split('.')
            if len(parts) >= 3:
                task = parts[1] if len(parts) > 1 else ''
                group_id = parts[2] if len(parts) > 2 else ''
                variant = parts[3] if len(parts) > 3 else ''
                temporal = parts[4] if len(parts) > 4 else ''
                visual = parts[5] if len(parts) > 5 else ''
                # P0-4: framing 解析（add_framing_probe.py 添加 .pos/.neu/.neg）
                framing = parts[6] if len(parts) > 6 else ''
            else:
                task = group_id = variant = temporal = visual = framing = ''

            # 解析模型回复 - 优先使用 pred 字段，其次使用 raw
            response = record.get('pred', record.get('raw', record.get('response', '')))
            pred_success = parse_verdict(response)

            # 模型名称可能在 meta.model 或顶层 model 字段
            model_name = record.get('model', record.get('meta', {}).get('model', 'unknown'))

            verdicts.append({
                'exam_id': exam_id,
                'model': model_name,
                'task': task,
                'group_id': group_id,
                'variant': variant,
                'temporal': temporal,
                'visual': visual,
                'framing': framing,  # P0-4: 新增，解析自 UID 7th part
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
        response = _clean_text(row.get("raw"))
        pred_val = row.get("pred")
        y_pred = _coerce_pred_label(pred_val, response)
        verdicts.append({
            "exam_id": _clean_text(row.get("uid")),
            "model": _clean_text(row.get("model")) or "unknown",
            "task": _clean_text(row.get("task")) or "unknown",
            "group_id": _clean_text(row.get("group_id")),
            "variant": _clean_text(row.get("variant")),
            "temporal": _clean_text(row.get("temporal")),
            "visual": _clean_text(row.get("visual")),
            "framing": _clean_text(row.get("framing")),
            "y_pred": y_pred,
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
    计算 IDR 指标
    
    Args:
        verdicts_df: 包含 variant, group_id, y_pred 的 DataFrame
        model: 可选，按模型筛选
    
    Returns:
        按模型分组的 IDR 结果字典
    """
    if model:
        verdicts_df = verdicts_df[verdicts_df['model'] == model].copy()
    
    results = {}
    
    # 按模型分组
    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}
        
        # 按任务分组
        for task, task_df in model_df.groupby('task'):
            # IDR only uses the baseline nuisance condition:
            # full.orig.clean.(no framing) vs cf.orig.clean.(no framing)
            full_df = _dedupe_group_rows(task_df, 'full').set_index('group_id')
            cf_df = _dedupe_group_rows(task_df, 'cf').set_index('group_id')
            
            # 找到共同的 group
            common_groups = full_df.index.intersection(cf_df.index)
            
            if len(common_groups) == 0:
                continue
            
            g_total = len(common_groups)
            
            # 计算 G_active（模型对 Full 判 Success 的 group）
            full_success = full_df.loc[common_groups, 'y_pred'] == 1
            g_active = int(full_success.sum())
            active_rate = g_active / g_total if g_total > 0 else 0.0
            
            # IDR_raw = P(ŷ_full=Success ∧ ŷ_cf=Fail)
            cf_fail = cf_df.loc[common_groups, 'y_pred'] == 0
            idr_raw = ((full_success) & (cf_fail)).sum() / g_total
            
            # IDR_cond = IDR_raw × G / G_active（仅在 Full 判 Success 的 group 上计算）
            if g_active > 0:
                idr_cond = idr_raw * g_total / g_active
            else:
                idr_cond = float('nan')
            
            # IDR_rev = P(ŷ_full=Fail ∧ ŷ_cf=Success)
            full_fail = full_df.loc[common_groups, 'y_pred'] == 0
            cf_success = cf_df.loc[common_groups, 'y_pred'] == 1
            idr_rev = ((full_fail) & (cf_success)).sum() / g_total
            
            # Degenerate 检测
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
    计算 Nuisance Stability (NS)

    NS_F = P(y_full_baseline = y_probe) 按 family 分组

    P0-4 fix: Added framing column filtering.
    - add_framing_probe.py generates framing variants (.pos/.neu/.neg) that have
      temporal=orig, visual=clean but different framing. These are FRAMING family probes,
      NOT Visual family probes. Including them in NS_vis corrupts the visual family metric.
    - Temporal family: temporal != orig (rev, blockswap_*), framing is ''
    - Visual family: visual != clean (noisy, style_*), framing is ''
    - Framing family: framing in (pos, neu, neg), temporal=orig, visual=clean

    Args:
        verdicts_df: 包含 variant, group_id, temporal, visual, framing, y_pred 的 DataFrame
        baseline_variant: 基准变体名称

    Returns:
        按模型/任务/family 分组的 NS 字典
    """
    results = {}

    for model_name, model_df in verdicts_df.groupby('model'):
        model_results = {}

        for task, task_df in model_df.groupby('task'):
            task_results = {}

            # 获取 baseline verdicts (variant=full 且 temporal=orig, visual=clean)
            baseline_df = _dedupe_group_rows(task_df, baseline_variant)
            baseline_by_group = dict(zip(baseline_df['group_id'], baseline_df['y_pred']))

            if not baseline_by_group:
                model_results[task] = task_results
                continue

            # 收集所有 probe 的 NS
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
    导出 NS-IDR 象限图所需的数据
    
    输出 CSV 包含: model, task, NS, IDR_raw, IDR_cond, G_active%, Degenerate
    """
    idr_results = compute_idr(verdicts_df)
    ns_results = compute_ns(verdicts_df)
    
    rows = []
    for model, model_idr in idr_results.items():
        model_ns = ns_results.get(model, {})
        for task, idr in model_idr.items():
            ns = model_ns.get(task, {})
            # 取平均 NS
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
    """打印 IDR 结果"""
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

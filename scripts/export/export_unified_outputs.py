#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export unified CSV outputs for GridWM-Judge runs.

Produces four tables from a scored exam run:
  - per-row.csv
  - group-level.csv
  - model-level.csv
  - figure-ready.csv

The exporter intentionally reuses the existing `score_exam.py` loading/scoring
logic so the CSV layer stays aligned with the SSOT scorer instead of drifting
into a parallel implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent  # repo root is 2 levels up
# Auto-add repo root so "scripts.xxx" imports work from any subdirectory
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from scripts.score.score_exam import (
    _default_weights_b,
    discover_latest_responses,
    load_gold_from_exam_dir,
    load_jsonl,
    load_responses,
    parse_task_b_json,
    score_one,
)

OUTPUT_SCHEMA_VERSION = "gridwm.output.v1"
TASK_FILES = ("task_a_exam.jsonl", "task_b_exam.jsonl", "task_c_exam.jsonl", "task_d_exam.jsonl")
TASK_E_FILES = ("task_e_exam_v13.jsonl", "task_e_exam_v12.jsonl", "task_e_exam_v11.jsonl", "task_e_exam_v1.jsonl")
TASK_C_VARIANTS = ("full", "nocue", "cf")
TASK_C_TEMPORALS = ("orig", "rev")
TASK_C_VISUALS = ("clean", "noisy", "style")
# Legacy Task C artifact (exams_pathb/) has full temporal/visual probe richness.
# Task D artifact (exams_taskc_repaired_candidate/) has only full/nocue/cf variants
# with temporal=orig, visual=clean — it does NOT contain temporal/visual probes.
TASK_D_VARIANTS = ("full", "nocue", "cf")


def _json_dumps(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _nonnull_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _mean_or_none(values: Iterable[Any]) -> Optional[float]:
    vals = [float(v) for v in values if pd.notna(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _bool_mean_or_none(values: Iterable[Any]) -> Optional[float]:
    vals = [bool(v) for v in values if pd.notna(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _sorted_join(values: Iterable[Any]) -> Optional[str]:
    items = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return "|".join(items) if items else None


def _parse_fail_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    fail_mask = df["parse_mode"].fillna("").astype(str).str.startswith("fail")
    return float(fail_mask.mean())


def _resolve_response_source(path: Path) -> Path:
    if path.is_file():
        return path
    direct_files = sorted(p for p in path.glob("*.jsonl") if p.is_file())
    if direct_files:
        return path
    # Handle request_bundle subdirectory structure (phase96-style):
    # e.g. request_bundle/e/composite/ or request_bundle/e/split/
    subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith("_")]
    if subdirs:
        for sd in subdirs:
            sd_files = sorted(p for p in sd.glob("*.jsonl") if p.is_file())
            if sd_files:
                return sd
    # Handle nested provider subdirectory (phase96 e_split style):
    # responses/openaicompatible_xxx/requests_taskE_split.jsonl
    nested_files = sorted(p for p in path.rglob("*.jsonl") if p.is_file())
    if nested_files:
        # Return the parent of the first match, for _response_files to glob from
        return nested_files[0].parent
    return discover_latest_responses(path)


def _response_files(path: Path) -> List[Path]:
    resolved = _resolve_response_source(path)
    if resolved.is_dir():
        return sorted(p for p in resolved.glob("*.jsonl") if p.is_file())
    return [resolved]


def _load_exam_records(exam_dir: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for name in TASK_FILES:
        path = exam_dir / name
        if not path.exists():
            continue
        for row in load_jsonl(path):
            uid = row.get("uid") or row.get("exam_id")
            if uid:
                records[str(uid)] = row
    # Task E exam files use versioned names; scan parent for any task_e_exam*.jsonl
    for name in TASK_E_FILES:
        path = exam_dir / name
        if path.exists():
            for row in load_jsonl(path):
                uid = row.get("uid") or row.get("exam_id")
                if uid:
                    records[str(uid)] = row
    return records


def _load_raw_response_records(path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for fp in _response_files(path):
        for row in load_jsonl(fp):
            uid = row.get("uid") or row.get("exam_id") or row.get("id")
            if uid and uid not in records:
                records[str(uid)] = row
    return records


def _infer_run_id(responses_path: Path, resolved_source: Path) -> str:
    if resolved_source.is_file():
        parent = resolved_source.parent
        return parent.name or resolved_source.stem
    if responses_path.is_dir():
        return responses_path.name
    return resolved_source.name


def _first_nonempty(records: Sequence[Dict[str, Any]], *path: str) -> Optional[Any]:
    for record in records:
        cur: Any = record
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur not in (None, ""):
            return cur
    return None


def _lookup_probe_row(df: pd.DataFrame, variant: str, temporal: str, visual: str) -> Optional[pd.Series]:
    subset = df[
        (df["variant"] == variant)
        & (df["temporal"] == temporal)
        & (df["visual"] == visual)
    ]
    if subset.empty:
        return None
    return subset.sort_values("uid").iloc[0]


def _lookup_variant_row(df: pd.DataFrame, variant: str) -> Optional[pd.Series]:
    """Look up a row by variant only (for Task D, which has no temporal/visual columns)."""
    subset = df[df["variant"] == variant]
    if subset.empty:
        return None
    return subset.sort_values("uid").iloc[0]


def _prediction_match(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[bool]:
    if a is None or b is None:
        return None
    pa = _nonnull_str(a.get("pred_norm"))
    pb = _nonnull_str(b.get("pred_norm"))
    if not pa or not pb:
        return None
    return pa == pb


def _task_c_group_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute legacy Task C group-level metrics.
    Only applicable to rows where task == "C" from exams_pathb/ artifact.
    Computes temporal/visual probe invariance (NS) and IDR from full/nocue/cf × temporal×visual panel.
    """
    out: Dict[str, Any] = {
        "c_panel_complete": False,
        "c_baseline_uid": None,
        "c_baseline_gold": None,
        "c_baseline_pred_norm": None,
        "c_baseline_correct": None,
        "c_full_acc": None,
        "c_nocue_acc": None,
        "c_cf_acc": None,
        "c_temporal_ns_full": None,
        "c_visual_ns_full": None,
        "c_idr_active": None,
        "c_idr_hit": None,
        "c_idr_rev": None,
    }
    # Guard: only run on genuine legacy Task C rows (task == "C"), NOT Task D
    if df.empty or str(df["task"].iloc[0]) != "C":
        return out

    expected_rows = len(TASK_C_VARIANTS) * len(TASK_C_TEMPORALS) * len(TASK_C_VISUALS)
    variant_ok = set(df["variant"].dropna()) >= set(TASK_C_VARIANTS)
    temporal_ok = set(df["temporal"].dropna()) >= set(TASK_C_TEMPORALS)
    visual_ok = set(df["visual"].dropna()) >= set(TASK_C_VISUALS)
    out["c_panel_complete"] = bool(len(df) >= expected_rows and variant_ok and temporal_ok and visual_ok)

    baseline = _lookup_probe_row(df, "full", "orig", "clean")
    if baseline is not None:
        out["c_baseline_uid"] = baseline.get("uid")
        out["c_baseline_gold"] = baseline.get("gold_answer_text")
        out["c_baseline_pred_norm"] = baseline.get("pred_norm")
        out["c_baseline_correct"] = baseline.get("correct")

    for variant in TASK_C_VARIANTS:
        subset = df[df["variant"] == variant]
        out[f"c_{variant}_acc"] = _mean_or_none(subset["correct"])

    temporal_matches: List[bool] = []
    for visual in TASK_C_VISUALS:
        orig = _lookup_probe_row(df, "full", "orig", visual)
        rev = _lookup_probe_row(df, "full", "rev", visual)
        match = _prediction_match(orig, rev)
        if match is not None:
            temporal_matches.append(match)
    out["c_temporal_ns_full"] = _bool_mean_or_none(temporal_matches)

    visual_matches: List[bool] = []
    baseline_clean = _lookup_probe_row(df, "full", "orig", "clean")
    for visual in ("noisy", "style"):
        probe = _lookup_probe_row(df, "full", "orig", visual)
        match = _prediction_match(baseline_clean, probe)
        if match is not None:
            visual_matches.append(match)
    out["c_visual_ns_full"] = _bool_mean_or_none(visual_matches)

    cf_clean = _lookup_probe_row(df, "cf", "orig", "clean")
    full_clean = baseline_clean
    if full_clean is not None and cf_clean is not None:
        full_pred = _nonnull_str(full_clean.get("pred_norm"))
        cf_pred = _nonnull_str(cf_clean.get("pred_norm"))
        if full_pred and cf_pred:
            out["c_idr_active"] = full_pred == "Success"
            out["c_idr_hit"] = full_pred == "Success" and cf_pred == "Fail"
            out["c_idr_rev"] = full_pred == "Fail" and cf_pred == "Success"

    return out


def _task_d_group_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute Task D group-level metrics (C4: causal intervention sensitivity).
    Only applicable to rows where task == "D" from exams_taskc_repaired_candidate/ artifact.
    Task D has no temporal/visual/framing probes — it measures continuous outcome judgment only.
    Primary metrics: correctness anchor + IDR (Full vs CF).
    """
    out: Dict[str, Any] = {
        "d_panel_complete": False,
        "d_baseline_uid": None,
        "d_baseline_gold": None,
        "d_baseline_pred_norm": None,
        "d_baseline_correct": None,
        "d_full_acc": None,
        "d_nocue_acc": None,
        "d_cf_acc": None,
        "d_jaccneu": None,
        "d_idr_active": None,
        "d_idr_hit": None,
        "d_idr_rev": None,
    }
    # Guard: only run on genuine Task D rows (task == "D"), NOT legacy Task C
    if df.empty or str(df["task"].iloc[0]) != "D":
        return out

    # Task D panel: 3 variants × 1 temporal × 1 visual = 3 rows per group
    variant_counts = df["variant"].value_counts()
    out["d_panel_complete"] = bool(
        len(df) >= 3
        and set(df["variant"].dropna()) >= set(TASK_D_VARIANTS)
    )

    # Baseline: full variant (JAccneu correctness anchor)
    # Task D has no temporal/visual probes — use variant-only lookup
    baseline = _lookup_variant_row(df, "full")
    if baseline is not None:
        out["d_baseline_uid"] = baseline.get("uid")
        out["d_baseline_gold"] = baseline.get("gold_answer_text")
        out["d_baseline_pred_norm"] = baseline.get("pred_norm")
        out["d_baseline_correct"] = baseline.get("correct")
        out["d_jaccneu"] = float(baseline.get("correct", False)) if pd.notna(baseline.get("correct")) else None

    # Per-variant accuracy
    for variant in TASK_D_VARIANTS:
        subset = df[df["variant"] == variant]
        out[f"d_{variant}_acc"] = _mean_or_none(subset["correct"])

    # IDR: full vs cf on a single group
    # Task D uses variant-only rows (no temporal/visual)
    full_row = _lookup_variant_row(df, "full")
    cf_row = _lookup_variant_row(df, "cf")
    if full_row is not None and cf_row is not None:
        full_pred = _nonnull_str(full_row.get("pred_norm"))
        cf_pred = _nonnull_str(cf_row.get("pred_norm"))
        if full_pred and cf_pred:
            out["d_idr_active"] = full_pred == "Success"
            out["d_idr_hit"] = full_pred == "Success" and cf_pred == "Fail"
            out["d_idr_rev"] = full_pred == "Fail" and cf_pred == "Success"

    return out


def _build_per_row_df(
    exam_dir: Path,
    responses_input: Path,
    run_id: Optional[str],
    model_id: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    resolved_source = _resolve_response_source(responses_input)
    gold = load_gold_from_exam_dir(exam_dir)
    responses, resp_telemetry = load_responses(resolved_source)
    raw_responses = _load_raw_response_records(resolved_source)
    exam_records = _load_exam_records(exam_dir)

    source_records = list(raw_responses.values())
    inferred_run_id = run_id or _infer_run_id(responses_input, resolved_source)
    inferred_model_id = model_id or _nonnull_str(
        _first_nonempty(source_records, "model")
        or _first_nonempty(source_records, "meta", "model")
        or inferred_run_id
    )
    backend = _nonnull_str(_first_nonempty(source_records, "meta", "backend"))
    provider = _nonnull_str(_first_nonempty(source_records, "meta", "provider"))

    weights_b = _default_weights_b()
    rows: List[Dict[str, Any]] = []
    for uid, gold_row in gold.items():
        resp = responses.get(uid)
        scored = score_one(uid, gold_row, resp, weights_b, b_acc_threshold=0.90)
        exam_record = exam_records.get(uid, {})
        raw_response = raw_responses.get(uid, {})
        meta = raw_response.get("meta", {}) if isinstance(raw_response, dict) else {}
        if not isinstance(meta, dict):
            meta = {}
        usage = meta.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}

        pred_json, _, _ = parse_task_b_json(raw_response.get("raw", raw_response.get("pred", ""))) if gold_row.task == "B" else (None, "", False)
        b_comp = scored.get("b_components", {}) if isinstance(scored.get("b_components"), dict) else {}

        if gold_row.task == "A":
            gold_answer_text = gold_row.a_answer
            gold_label = gold_row.a_label
            gold_answer_json = None
        elif gold_row.task == "B":
            gold_answer_text = exam_record.get("answer")
            gold_label = None
            gold_answer_json = gold_row.b_answer_json
        else:
            gold_answer_text = gold_row.c_answer
            gold_label = gold_row.c_label
            gold_answer_json = None

        row = {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "per_row",
            "run_id": inferred_run_id,
            "model_id": inferred_model_id,
            "backend": backend,
            "provider": provider,
            "source_exam_dir": str(exam_dir),
            "source_responses": str(resolved_source),
            "uid": uid,
            "exam_id": raw_response.get("exam_id", exam_record.get("exam_id", uid)),
            "task": scored.get("task"),
            "env_task": scored.get("env_task"),
            "group_id": scored.get("group_id"),
            "t": scored.get("t"),
            "variant": scored.get("variant"),
            "temporal": scored.get("temporal"),
            "visual": scored.get("visual"),
            "framing": exam_record.get("framing"),
            "image_relpath": exam_record.get("image"),
            "prompt": exam_record.get("prompt"),
            "gold_answer_text": gold_answer_text,
            "gold_label": gold_label,
            "gold_answer_json": _json_dumps(gold_answer_json),
            "response_text_raw": raw_response.get("raw", resp.raw if resp else None),
            "response_text_pred": raw_response.get("pred", resp.pred if resp else None),
            "pred_norm": scored.get("pred_norm"),
            "pred_answer_json": _json_dumps(pred_json),
            "score": float(scored.get("score", 0.0)),
            "correct": bool(scored.get("correct", False)),
            "failure": scored.get("failure"),
            "parse_mode": scored.get("parse_mode"),
            "api_ok": bool(resp.ok) if resp is not None else False,
            "api_error": resp.error if resp is not None else None,
            "latency_ms": meta.get("latency_ms"),
            "finish_reason": meta.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "exam_schema_version": exam_record.get("schema_version", gold_row.exam_schema_version),
            "response_schema_version": raw_response.get("schema_version", resp.schema_version if resp else None),
            "b_component_agent_pos": b_comp.get("agent_pos"),
            "b_component_agent_dir": b_comp.get("agent_dir"),
            "b_component_carrying": b_comp.get("carrying"),
            "b_component_front_cell": b_comp.get("front_cell"),
            "b_component_objects_jaccard": b_comp.get("objects_jaccard"),
            "b_n_gold_objects": b_comp.get("n_gold_objects"),
            "b_n_pred_objects": b_comp.get("n_pred_objects"),
            "b_acc_threshold": scored.get("b_acc_threshold"),
            "b_acc_at_threshold": scored.get("b_acc_at_threshold"),
            "has_markdown_fence": scored.get("has_markdown_fence"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["task", "env_task", "group_id", "uid"]).reset_index(drop=True)
    meta = {
        "run_id": inferred_run_id,
        "model_id": inferred_model_id,
        "backend": backend,
        "provider": provider,
        "resolved_responses": str(resolved_source),
        "response_telemetry": resp_telemetry,
    }
    return df, meta


def _build_group_level_df(per_row_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["run_id", "model_id", "backend", "provider", "task", "env_task", "group_id"]
    for keys, group in per_row_df.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys))
        row = {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "group_level",
            **key_map,
            "n_rows": int(len(group)),
            "n_correct": int(group["correct"].sum()),
            "acc": float(group["correct"].mean()),
            "mean_score": float(group["score"].mean()),
            "api_error_n": int((group["failure"] == "api_error").sum()),
            "missing_n": int((group["failure"] == "missing_response").sum()),
            "parse_fail_n": int(group["parse_mode"].fillna("").astype(str).str.startswith("fail").sum()),
            "variant_set": _sorted_join(group["variant"]),
            "temporal_set": _sorted_join(group["temporal"]),
            "visual_set": _sorted_join(group["visual"]),
        }
        row.update(_task_c_group_metrics(group))
        row.update(_task_d_group_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["task", "env_task", "group_id"]).reset_index(drop=True)


def _slice_row(
    per_row_df: pd.DataFrame,
    group_level_df: pd.DataFrame,
    run_id: str,
    model_id: str,
    backend: Optional[str],
    provider: Optional[str],
    task: str,
    env_task: str,
) -> Dict[str, Any]:
    subset = per_row_df.copy()
    if task != "ALL":
        subset = subset[subset["task"] == task]
    if env_task != "ALL":
        subset = subset[subset["env_task"] == env_task]

    c_group_subset = group_level_df[group_level_df["task"] == "C"].copy()
    if env_task != "ALL":
        c_group_subset = c_group_subset[c_group_subset["env_task"] == env_task]
    if task not in ("ALL", "C"):
        c_group_subset = c_group_subset.iloc[0:0]

    d_group_subset = group_level_df[group_level_df["task"] == "D"].copy()
    if env_task != "ALL":
        d_group_subset = d_group_subset[d_group_subset["env_task"] == env_task]
    if task not in ("ALL", "D"):
        d_group_subset = d_group_subset.iloc[0:0]

    slice_kind = "overall"
    if task != "ALL" and env_task == "ALL":
        slice_kind = "task"
    elif task != "ALL" and env_task != "ALL":
        slice_kind = "task_env"

    row = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "aggregation_level": "model_level",
        "slice_kind": slice_kind,
        "run_id": run_id,
        "model_id": model_id,
        "backend": backend,
        "provider": provider,
        "task": task,
        "env_task": env_task,
        "n_rows": int(len(subset)),
        "n_groups": int(subset["group_id"].nunique()) if not subset.empty else 0,
        "n_correct": int(subset["correct"].sum()) if not subset.empty else 0,
        "acc": float(subset["correct"].mean()) if not subset.empty else 0.0,
        "mean_score": float(subset["score"].mean()) if not subset.empty else 0.0,
        "api_error_rate": float((subset["failure"] == "api_error").mean()) if not subset.empty else 0.0,
        "missing_rate": float((subset["failure"] == "missing_response").mean()) if not subset.empty else 0.0,
        "parse_fail_rate": _parse_fail_rate(subset),
        "c_groups_complete": int(c_group_subset["c_panel_complete"].fillna(False).sum()) if not c_group_subset.empty else 0,
        "c_full_acc_mean": _mean_or_none(c_group_subset["c_full_acc"]) if not c_group_subset.empty else None,
        "c_nocue_acc_mean": _mean_or_none(c_group_subset["c_nocue_acc"]) if not c_group_subset.empty else None,
        "c_cf_acc_mean": _mean_or_none(c_group_subset["c_cf_acc"]) if not c_group_subset.empty else None,
        "c_temporal_ns_full_mean": _mean_or_none(c_group_subset["c_temporal_ns_full"]) if not c_group_subset.empty else None,
        "c_visual_ns_full_mean": _mean_or_none(c_group_subset["c_visual_ns_full"]) if not c_group_subset.empty else None,
        "c_idr_active_rate_mean": _bool_mean_or_none(c_group_subset["c_idr_active"]) if not c_group_subset.empty else None,
        "c_idr_hit_rate_mean": _bool_mean_or_none(c_group_subset["c_idr_hit"]) if not c_group_subset.empty else None,
        "c_idr_rev_rate_mean": _bool_mean_or_none(c_group_subset["c_idr_rev"]) if not c_group_subset.empty else None,
        "d_groups_complete": int(d_group_subset["d_panel_complete"].fillna(False).sum()) if not d_group_subset.empty else 0,
        "d_full_acc_mean": _mean_or_none(d_group_subset["d_full_acc"]) if not d_group_subset.empty else None,
        "d_nocue_acc_mean": _mean_or_none(d_group_subset["d_nocue_acc"]) if not d_group_subset.empty else None,
        "d_cf_acc_mean": _mean_or_none(d_group_subset["d_cf_acc"]) if not d_group_subset.empty else None,
        "d_jaccneu_mean": _mean_or_none(d_group_subset["d_jaccneu"]) if not d_group_subset.empty else None,
        "d_idr_active_rate_mean": _bool_mean_or_none(d_group_subset["d_idr_active"]) if not d_group_subset.empty else None,
        "d_idr_hit_rate_mean": _bool_mean_or_none(d_group_subset["d_idr_hit"]) if not d_group_subset.empty else None,
        "d_idr_rev_rate_mean": _bool_mean_or_none(d_group_subset["d_idr_rev"]) if not d_group_subset.empty else None,
        "b_component_agent_pos_mean": _mean_or_none(subset["b_component_agent_pos"]) if not subset.empty else None,
        "b_component_agent_dir_mean": _mean_or_none(subset["b_component_agent_dir"]) if not subset.empty else None,
        "b_component_carrying_mean": _mean_or_none(subset["b_component_carrying"]) if not subset.empty else None,
        "b_component_front_cell_mean": _mean_or_none(subset["b_component_front_cell"]) if not subset.empty else None,
        "b_component_objects_jaccard_mean": _mean_or_none(subset["b_component_objects_jaccard"]) if not subset.empty else None,
    }
    return row


def _build_model_level_df(per_row_df: pd.DataFrame, group_level_df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    run_id = meta["run_id"]
    model_id = meta["model_id"]
    backend = meta.get("backend")
    provider = meta.get("provider")

    rows: List[Dict[str, Any]] = []
    rows.append(_slice_row(per_row_df, group_level_df, run_id, model_id, backend, provider, "ALL", "ALL"))
    for task in ("A", "B", "C", "D"):
        rows.append(_slice_row(per_row_df, group_level_df, run_id, model_id, backend, provider, task, "ALL"))
        task_envs = sorted(v for v in per_row_df.loc[per_row_df["task"] == task, "env_task"].dropna().unique())
        for env_task in task_envs:
            rows.append(_slice_row(per_row_df, group_level_df, run_id, model_id, backend, provider, task, str(env_task)))
    return pd.DataFrame(rows)


def _build_figure_ready_df(per_row_df: pd.DataFrame, group_level_df: pd.DataFrame, model_level_df: pd.DataFrame) -> pd.DataFrame:
    if per_row_df.empty:
        return pd.DataFrame()

    meta_cols = {
        "run_id": per_row_df["run_id"].iloc[0],
        "model_id": per_row_df["model_id"].iloc[0],
        "backend": per_row_df["backend"].iloc[0],
        "provider": per_row_df["provider"].iloc[0],
    }
    rows: List[Dict[str, Any]] = []

    task_env_acc = model_level_df[
        (model_level_df["aggregation_level"] == "model_level")
        & (model_level_df["slice_kind"] == "task_env")
    ]
    for _, row in task_env_acc.iterrows():
        rows.append({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "figure_ready",
            "plot_id": "task_env_accuracy",
            "geom": "bar",
            **meta_cols,
            "task": row["task"],
            "env_task": row["env_task"],
            "group_id": None,
            "variant": None,
            "temporal": None,
            "visual": None,
            "x": row["env_task"],
            "y": None,
            "series": row["task"],
            "label": None,
            "metric": "acc",
            "value": row["acc"],
            "n": row["n_rows"],
        })

    c_variant = (
        per_row_df[per_row_df["task"] == "C"]
        .groupby(["env_task", "variant"], dropna=False)
        .agg(value=("correct", "mean"), n=("uid", "count"))
        .reset_index()
    )
    for _, row in c_variant.iterrows():
        rows.append({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "figure_ready",
            "plot_id": "task_c_variant_accuracy",
            "geom": "bar",
            **meta_cols,
            "task": "C",
            "env_task": row["env_task"],
            "group_id": None,
            "variant": row["variant"],
            "temporal": None,
            "visual": None,
            "x": row["variant"],
            "y": None,
            "series": row["env_task"],
            "label": None,
            "metric": "acc",
            "value": row["value"],
            "n": row["n"],
        })

    c_probe = (
        per_row_df[per_row_df["task"] == "C"]
        .groupby(["env_task", "variant", "temporal", "visual"], dropna=False)
        .agg(value=("correct", "mean"), n=("uid", "count"))
        .reset_index()
    )
    for _, row in c_probe.iterrows():
        probe_key = ".".join(str(row[col]) for col in ("variant", "temporal", "visual"))
        rows.append({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "figure_ready",
            "plot_id": "task_c_probe_accuracy",
            "geom": "bar",
            **meta_cols,
            "task": "C",
            "env_task": row["env_task"],
            "group_id": None,
            "variant": row["variant"],
            "temporal": row["temporal"],
            "visual": row["visual"],
            "x": probe_key,
            "y": None,
            "series": row["env_task"],
            "label": None,
            "metric": "acc",
            "value": row["value"],
            "n": row["n"],
        })

    # Task D: variant accuracy (full/nocue/cf) — Task D primary correctness decomposition
    d_variant = (
        per_row_df[per_row_df["task"] == "D"]
        .groupby(["env_task", "variant"], dropna=False)
        .agg(value=("correct", "mean"), n=("uid", "count"))
        .reset_index()
    )
    for _, row in d_variant.iterrows():
        rows.append({
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "aggregation_level": "figure_ready",
            "plot_id": "task_d_variant_accuracy",
            "geom": "bar",
            **meta_cols,
            "task": "D",
            "env_task": row["env_task"],
            "group_id": None,
            "variant": row["variant"],
            "temporal": None,
            "visual": None,
            "x": row["variant"],
            "y": None,
            "series": row["env_task"],
            "label": None,
            "metric": "acc",
            "value": row["value"],
            "n": row["n"],
        })

    # Task D: NS-IDR scatter per group (NS from legacy C; IDR from Task D)
    # This plot joins legacy C NS with Task D IDR for the NS-IDR quadrant figure.
    # Only produces rows when legacy C NS data and Task D IDR data are both present
    # for the same (env_task, group_id) combination.
    c_group_ns = group_level_df[group_level_df["task"] == "C"][["env_task", "group_id", "c_temporal_ns_full", "c_visual_ns_full"]].copy()
    d_group_idr = group_level_df[group_level_df["task"] == "D"][["env_task", "group_id", "d_idr_hit"]].copy()
    ns_idr_joined = c_group_ns.merge(d_group_idr, on=["env_task", "group_id"], how="inner")
    for _, row in ns_idr_joined.iterrows():
        for ns_col, ns_name in [("c_temporal_ns_full", "temporal"), ("c_visual_ns_full", "visual")]:
            if pd.notna(row.get(ns_col)) and pd.notna(row.get("d_idr_hit")):
                rows.append({
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "aggregation_level": "figure_ready",
                    "plot_id": f"ns_idr_scatter_{ns_name}",
                    "geom": "point",
                    **meta_cols,
                    "task": "C+D",
                    "env_task": row["env_task"],
                    "group_id": row["group_id"],
                    "variant": None,
                    "temporal": None,
                    "visual": None,
                    "x": float(row["d_idr_hit"]),
                    "y": float(row[ns_col]),
                    "series": row["env_task"],
                    "label": row["group_id"],
                    "metric": None,
                    "value": None,
                    "n": 1,
                })

    b_rows = per_row_df[per_row_df["task"] == "B"]
    if not b_rows.empty:
        component_map = {
            "b_component_agent_pos": "agent_pos",
            "b_component_agent_dir": "agent_dir",
            "b_component_carrying": "carrying",
            "b_component_front_cell": "front_cell",
            "b_component_objects_jaccard": "objects_jaccard",
        }
        for column, component in component_map.items():
            comp_df = (
                b_rows.groupby("env_task", dropna=False)[column]
                .mean()
                .reset_index(name="value")
            )
            for _, row in comp_df.iterrows():
                rows.append({
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "aggregation_level": "figure_ready",
                    "plot_id": "task_b_component_mean",
                    "geom": "bar",
                    **meta_cols,
                    "task": "B",
                    "env_task": row["env_task"],
                    "group_id": None,
                    "variant": None,
                    "temporal": None,
                    "visual": None,
                    "x": component,
                    "y": None,
                    "series": row["env_task"],
                    "label": None,
                    "metric": "mean_component_score",
                    "value": row["value"],
                    "n": int((b_rows["env_task"] == row["env_task"]).sum()),
                })

    c_groups = group_level_df[group_level_df["task"] == "C"]
    for _, row in c_groups.iterrows():
        if pd.notna(row.get("c_idr_hit")) and pd.notna(row.get("c_visual_ns_full")):
            rows.append({
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "aggregation_level": "figure_ready",
                "plot_id": "task_c_group_scatter_visual",
                "geom": "point",
                **meta_cols,
                "task": "C",
                "env_task": row["env_task"],
                "group_id": row["group_id"],
                "variant": None,
                "temporal": None,
                "visual": None,
                "x": float(row["c_idr_hit"]),
                "y": float(row["c_visual_ns_full"]),
                "series": row["env_task"],
                "label": row["group_id"],
                "metric": None,
                "value": None,
                "n": 1,
            })
        if pd.notna(row.get("c_idr_hit")) and pd.notna(row.get("c_temporal_ns_full")):
            rows.append({
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "aggregation_level": "figure_ready",
                "plot_id": "task_c_group_scatter_temporal",
                "geom": "point",
                **meta_cols,
                "task": "C",
                "env_task": row["env_task"],
                "group_id": row["group_id"],
                "variant": None,
                "temporal": None,
                "visual": None,
                "x": float(row["c_idr_hit"]),
                "y": float(row["c_temporal_ns_full"]),
                "series": row["env_task"],
                "label": row["group_id"],
                "metric": None,
                "value": None,
                "n": 1,
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unified CSV outputs for GridWM-Judge runs")
    parser.add_argument("--exam_dir", required=True, help="Path to exam directory")
    parser.add_argument("--responses", required=True, help="Path to responses JSONL or directory")
    parser.add_argument("--out_dir", required=True, help="Directory for CSV outputs")
    parser.add_argument("--run_id", default=None, help="Optional run id override")
    parser.add_argument("--model_id", default=None, help="Optional model id override")
    args = parser.parse_args()

    exam_dir = Path(args.exam_dir).resolve()
    responses_input = Path(args.responses).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row_df, meta = _build_per_row_df(
        exam_dir=exam_dir,
        responses_input=responses_input,
        run_id=args.run_id,
        model_id=args.model_id,
    )
    group_level_df = _build_group_level_df(per_row_df)
    model_level_df = _build_model_level_df(per_row_df, group_level_df, meta)
    figure_ready_df = _build_figure_ready_df(per_row_df, group_level_df, model_level_df)

    per_row_path = out_dir / "per-row.csv"
    group_level_path = out_dir / "group-level.csv"
    model_level_path = out_dir / "model-level.csv"
    figure_ready_path = out_dir / "figure-ready.csv"
    manifest_path = out_dir / "export_manifest.json"

    per_row_df.to_csv(per_row_path, index=False)
    group_level_df.to_csv(group_level_path, index=False)
    model_level_df.to_csv(model_level_path, index=False)
    figure_ready_df.to_csv(figure_ready_path, index=False)

    manifest = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "exam_dir": str(exam_dir),
        "responses": meta["resolved_responses"],
        "run_id": meta["run_id"],
        "model_id": meta["model_id"],
        "backend": meta.get("backend"),
        "provider": meta.get("provider"),
        "counts": {
            "per_row": int(len(per_row_df)),
            "group_level": int(len(group_level_df)),
            "model_level": int(len(model_level_df)),
            "figure_ready": int(len(figure_ready_df)),
        },
        "response_telemetry": meta.get("response_telemetry", {}),
        "outputs": {
            "per_row": str(per_row_path),
            "group_level": str(group_level_path),
            "model_level": str(model_level_path),
            "figure_ready": str(figure_ready_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Exported unified outputs to {out_dir}")
    print(f"  per-row.csv      {len(per_row_df):>6} rows")
    print(f"  group-level.csv  {len(group_level_df):>6} rows")
    print(f"  model-level.csv  {len(model_level_df):>6} rows")
    print(f"  figure-ready.csv {len(figure_ready_df):>6} rows")


if __name__ == "__main__":
    main()

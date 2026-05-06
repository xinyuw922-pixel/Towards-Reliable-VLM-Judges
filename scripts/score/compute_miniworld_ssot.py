#!/usr/bin/env python3
"""
compute_miniworld_ssot.py

Compute SSOT metrics for MiniWorld Composite representation from the
closeout_writer_bundle_2026-04-09 bundle.

Data source: paper_exports/closeout_writer_bundle_2026-04-09/miniworld_closeout/

Task C column mapping:
  - score_per_row has: uid, family, template_id, variant, framing, gold, pred, correct, ...
  - group_id = family.template_id (use family as proxy since template_id alone isn't unique)
  - For NS_frame: pivot on (family+template_id, variant, framing)

Task D column mapping:
  - score_per_row has: uid, family, query_template_id, query_variant, framing, gold, pred, correct
  - group_id = family + query_template_id
  - variant = query_variant

Metrics:
  - Acc_A1:     Task A 4-choice accuracy
  - SuccC:      Task C base rate of "Success" verdicts (full variant)
  - Acc_C_base: Task C full-variant accuracy (deduplicated by group_id)
  - Acc_nocue / Acc_cf: Task C nocue/cf accuracy
  - NS_frame:   Noise sensitivity under framing (flip rate across pos/neu/neg pairs)
  - JAccneu:    Task D full-variant accuracy
  - Acc_D_nocue / Acc_D_cf: Task D variant accuracy
  - IDR_raw/cond: Interventional Discrimination Rate (from Task D, fallback to Task C)
  - G_active_pct: Proportion of groups where not all three variants agree

Notes:
  - Task D base has only full/cf variants (no nocue) — cannot compute 3-way IDR.
  - Therefore IDR is ALWAYS computed from Task C base (which has full/nocue/cf triplets
    for both models) for consistency.
  - JAccneu: from Task D base when available (gemini-2.5-flash); from Task C base
    for gpt-5.4 (no Task D base), marked as "taskC_fallback".
  - Acc_D_nocue/cf: only available for gemini-2.5-flash (has Task D base data).
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import sys as _sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

BUNDLE = _ROOT / "paper_exports" / "closeout_writer_bundle_2026-04-09" / "miniworld_closeout"

MODEL_CONFIGS = {
    "gpt-5.4": {
        "task_a":      BUNDLE / "taska" / "responses_full_gpt54_responses" / "score_per_row_gpt54_crossenv.jsonl",
        "taskc_base":  BUNDLE / "taskc" / "responses_full_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses" / "score_per_row.jsonl",
        "taskc_frame":  BUNDLE / "taskc" / "framing_full" / "responses_full_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses" / "score_per_row.jsonl",
        "taskd_base":  None,
        "taskd_frame":  BUNDLE / "taskd" / "framing_full" / "responses_full_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses" / "score_per_row.jsonl",
    },
    "gemini-2.5-flash": {
        "task_a":      BUNDLE / "taska" / "responses_full_gemini25flash_chat" / "score_per_row_gemini25flash_crossenv.jsonl",
        "taskc_base":  BUNDLE / "taskc" / "responses_full_gemini25flash_chat" / "openaicompatible_zhizengzeng_gemini-2.5-flash_chat" / "score_per_row.jsonl",
        "taskc_frame":  BUNDLE / "taskc" / "framing_full" / "responses_full_gemini25flash_chat" / "openaicompatible_zhizengzeng_gemini-2.5-flash_chat" / "score_per_row.jsonl",
        "taskd_base":  BUNDLE / "taskd" / "responses_full_gemini25flash_chat" / "openaicompatible_zhizengzeng_gemini-2.5-flash_chat" / "score_per_row.jsonl",
        "taskd_frame":  BUNDLE / "taskd" / "framing_full" / "responses_full_gemini25flash_chat" / "openaicompatible_zhizengzeng_gemini-2.5-flash_chat" / "score_per_row.jsonl",
    },
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load(path: Optional[Path], skip_unknown: bool = True) -> pd.DataFrame:
    """Load score_per_row JSONL, skip unknown/error rows."""
    if path is None or not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if skip_unknown and d.get("unknown"):
                continue
            rows.append(d)
    return pd.DataFrame(rows)


def _ensure_group_id_c(df: pd.DataFrame) -> pd.DataFrame:
    """Add group_id column to Task C DataFrame.

    UID pattern: C-MW.<family>.<template>.<seed>.<variant>[.<framing>]
    parts[0]="C-MW", parts[1]=<family>, parts[2]=<template>, parts[3]=<seed>,
    parts[4]=<variant>, parts[5]=<framing> (optional)
    group_id = <template>.<seed> = parts[2].parts[3]
    """
    if df.empty:
        return df
    if "group_id" in df.columns:
        return df

    def make_gid(uid: str) -> str:
        parts = uid.split(".")
        # "C-MW.<family>.<template>.<seed>.<variant>[.<framing>]"
        if len(parts) >= 4 and parts[0] == "C-MW":
            return f"{parts[2]}.{parts[3]}"   # template.seed
        return uid

    df = df.copy()
    df["group_id"] = df["uid"].apply(make_gid)
    return df


def _ensure_group_id_d(df: pd.DataFrame) -> pd.DataFrame:
    """Add group_id column to Task D DataFrame."""
    if df.empty:
        return df
    if "group_id" in df.columns:
        return df
    # group_id = family.query_template_id
    def make_gid(row: Dict[str, Any]) -> str:
        return f"{row.get('family','')}.{row.get('query_template_id','')}"
    df = df.copy()
    df["group_id"] = df.apply(make_gid, axis=1)
    return df


def _ensure_variant_d(df: pd.DataFrame) -> pd.DataFrame:
    """Rename query_variant -> variant for Task D DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    if "query_variant" in df.columns and "variant" not in df.columns:
        df["variant"] = df["query_variant"]
    return df


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------

def _acc(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    valid = df[df["correct"].notna()]
    if valid.empty:
        return None
    return float(valid["correct"].mean())


def _n(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    return len(df)


def _succ_rate(df: Optional[pd.DataFrame]) -> Optional[float]:
    """Proportion of rows where model predicts 'Success'."""
    if df is None or df.empty:
        return None
    valid = df[df["pred"].notna()]
    if valid.empty:
        return None
    return float((valid["pred"] == "Success").mean())


def _ns_frame(df: Optional[pd.DataFrame]) -> Optional[float]:
    """
    NS_frame: framing noise sensitivity.
    Average flip rate across (pos,neu), (pos,neg), (neu,neg) pairs.
    Uses full variant only, pivoted on group_id.
    Uses groupby+unstack to handle potential duplicate rows.
    """
    if df is None or df.empty:
        return None
    df = _ensure_group_id_c(df)
    sub = df[df["variant"] == "full"].dropna(subset=["group_id", "framing", "pred"])
    if sub.empty:
        return None

    try:
        pivot = (
            sub.drop_duplicates(["group_id", "framing"])
            .groupby(["group_id", "framing"])["pred"]
            .first()
            .unstack("framing")
        )
    except Exception:
        return None

    pairs = [("pos", "neu"), ("pos", "neg"), ("neu", "neg")]
    rates = []
    for a, b in pairs:
        if a in pivot.columns and b in pivot.columns:
            both = pivot[[a, b]].dropna()
            if len(both) > 0:
                rates.append(float((both[a] != both[b]).mean()))
    if not rates:
        return None
    return float(sum(rates) / len(rates))


def _ns_frame_d(df: Optional[pd.DataFrame]) -> Optional[float]:
    """NS_frame for Task D framing_full. Uses groupby+unstack."""
    if df is None or df.empty:
        return None
    df = _ensure_group_id_d(df)
    df = _ensure_variant_d(df)
    sub = df[df["variant"] == "full"].dropna(subset=["group_id", "framing", "pred"])
    if sub.empty:
        return None

    try:
        pivot = (
            sub.drop_duplicates(["group_id", "framing"])
            .groupby(["group_id", "framing"])["pred"]
            .first()
            .unstack("framing")
        )
    except Exception:
        return None

    pairs = [("pos", "neu"), ("pos", "neg"), ("neu", "neg")]
    rates = []
    for a, b in pairs:
        if a in pivot.columns and b in pivot.columns:
            both = pivot[[a, b]].dropna()
            if len(both) > 0:
                rates.append(float((both[a] != both[b]).mean()))
    if not rates:
        return None
    return float(sum(rates) / len(rates))


def _compute_idr_d(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute IDR on Task D base data. Uses groupby+unstack.
    All three variants (full/nocue/cf) must be present.
    """
    df = _ensure_group_id_d(df)
    df = _ensure_variant_d(df)

    available = set(df["variant"].unique())
    if available != {"full", "nocue", "cf"}:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    df = df.copy()
    df["y_pred"] = df["pred"].apply(
        lambda p: 1 if str(p) == "Success" else 0 if str(p) == "Fail" else -1
    )
    df = df[df["y_pred"] >= 0]

    try:
        pivot = (
            df.groupby(["group_id", "variant"])["y_pred"]
            .first()
            .unstack("variant")
        )
    except Exception:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    pivot = pivot.dropna()
    if pivot.empty:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    n_groups = len(pivot)
    all_same = (
        (pivot["full"] == pivot["nocue"]) & (pivot["nocue"] == pivot["cf"])
    )
    idr_raw = float(all_same.mean())
    g_active = float((~all_same).mean())

    active = pivot[~all_same]
    idr_cond = (
        float(
            ((active["full"] == active["nocue"]) & (active["nocue"] == active["cf"])).mean()
        )
        if not active.empty
        else None
    )

    return dict(IDR_raw=idr_raw, IDR_cond=idr_cond, G_active_pct=g_active, n_groups=n_groups)


def _compute_idr_c(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Compute IDR on Task C base data. Uses full/nocue/cf triplet.
    Uses groupby+unstack to avoid pivot duplicate-index issues.
    """
    df = _ensure_group_id_c(df)

    # Check all three variants are present
    available = set(df["variant"].unique())
    if available != {"full", "nocue", "cf"}:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    # Compute y_pred
    df = df.copy()
    df["y_pred"] = df["pred"].apply(
        lambda p: 1 if str(p) == "Success" else 0 if str(p) == "Fail" else -1
    )
    df = df[df["y_pred"] >= 0]

    # groupby+unstack avoids duplicate-row issues in pivot()
    try:
        pivot = (
            df.groupby(["group_id", "variant"])["y_pred"]
            .first()
            .unstack("variant")
        )
    except Exception:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    pivot = pivot.dropna()
    if pivot.empty:
        return dict(IDR_raw=None, IDR_cond=None, G_active_pct=None, n_groups=0)

    n_groups = len(pivot)
    all_same = (
        (pivot["full"] == pivot["nocue"]) & (pivot["nocue"] == pivot["cf"])
    )
    idr_raw = float(all_same.mean())
    g_active = float((~all_same).mean())

    active = pivot[~all_same]
    idr_cond = (
        float(
            ((active["full"] == active["nocue"]) & (active["nocue"] == active["cf"])).mean()
        )
        if not active.empty
        else None
    )

    return dict(IDR_raw=idr_raw, IDR_cond=idr_cond, G_active_pct=g_active, n_groups=n_groups)


# ---------------------------------------------------------------------------
# Per-model computation
# ---------------------------------------------------------------------------

def compute_model(model: str, cfg: Dict[str, Path]) -> Dict[str, Any]:
    r = {"model": model}

    df_a     = _load(cfg["task_a"])
    df_cb    = _load(cfg["taskc_base"])
    df_cf    = _load(cfg["taskc_frame"])
    df_db    = _load(cfg["taskd_base"])
    df_df    = _load(cfg["taskd_frame"])

    # Normalize group_id and variant for each DataFrame
    df_cb = _ensure_group_id_c(df_cb)
    df_cf = _ensure_group_id_c(df_cf)
    df_db = _ensure_group_id_d(_ensure_variant_d(df_db))
    df_df = _ensure_group_id_d(_ensure_variant_d(df_df))

    # ---- Task A ----
    r["Acc_A1"] = _acc(df_a)
    r["n_A"]    = _n(df_a)

    # ---- Task C base ----
    if not df_cb.empty:
        c_full  = df_cb[df_cb["variant"] == "full"].drop_duplicates("group_id")
        c_nocue = df_cb[df_cb["variant"] == "nocue"].drop_duplicates("group_id")
        c_cf    = df_cb[df_cb["variant"] == "cf"].drop_duplicates("group_id")

        r["Acc_C_base"]  = _acc(c_full)
        r["Acc_nocue"]   = _acc(c_nocue)
        r["Acc_cf"]      = _acc(c_cf)
        r["SuccC"]       = _succ_rate(c_full)
        r["n_C_groups"]  = len(c_full)
    else:
        r["Acc_C_base"] = r["Acc_nocue"] = r["Acc_cf"] = r["SuccC"] = None
        r["n_C_groups"] = 0

    # ---- Framing metrics ----
    r["NS_frame"]  = _ns_frame(df_cf)     # Task C framing_full
    r["NS_frame_D"] = _ns_frame_d(df_df)  # Task D framing_full
    r["n_C_frame"] = _n(df_cf)
    r["n_D_frame"] = _n(df_df)

    # ---- JAccneu / IDR ----
    # IDR always from Task C base (has full/nocue/cf triplets for both models)
    if not df_cb.empty:
        c_full = df_cb[df_cb["variant"] == "full"].drop_duplicates("group_id")
        r["JAccneu"] = _acc(c_full)   # fallback (Task C full) if no Task D base
        idr = _compute_idr_c(df_cb)
        r["IDR_raw"]      = idr["IDR_raw"]
        r["IDR_cond"]     = idr["IDR_cond"]
        r["G_active_pct"] = idr["G_active_pct"]
        r["n_IDR_groups"] = idr["n_groups"]
        r["IDR_source"]   = "taskC_base"
    else:
        r["JAccneu"] = None
        r["IDR_raw"] = r["IDR_cond"] = r["G_active_pct"] = None
        r["n_IDR_groups"] = 0
        r["IDR_source"] = "unavailable"

    # Task D-specific metrics (only if Task D base available)
    if not df_db.empty:
        d_full  = df_db[df_db["variant"] == "full"].drop_duplicates("group_id")
        d_nocue = df_db[df_db["variant"] == "nocue"].drop_duplicates("group_id")
        d_cf    = df_db[df_db["variant"] == "cf"].drop_duplicates("group_id")
        r["JAccneu_D"]     = _acc(d_full)
        r["Acc_D_nocue"]   = _acc(d_nocue)
        r["Acc_D_cf"]      = _acc(d_cf)
        r["n_D_groups"]    = len(d_full)
    else:
        r["JAccneu_D"]   = None
        r["Acc_D_nocue"] = None
        r["Acc_D_cf"]    = None
        r["n_D_groups"]  = 0

    return r


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "NA"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


METRIC_COLS = [
    "model",
    "Acc_A1",
    "SuccC",
    "Acc_C_base",
    "Acc_nocue",
    "Acc_cf",
    "NS_frame",
    "NS_frame_D",
    "JAccneu",
    "JAccneu_D",
    "Acc_D_nocue",
    "Acc_D_cf",
    "IDR_raw",
    "IDR_cond",
    "G_active_pct",
    "IDR_source",
]

METRIC_COLS_RAW = METRIC_COLS + [
    "n_A", "n_C_groups", "n_C_frame", "n_D_groups", "n_D_frame", "n_IDR_groups",
]

KEY_LINES = [
    ("Acc_A1",       "Task A 4-choice accuracy (all rows)"),
    ("SuccC",        "Task C full: rate of 'Success' verdicts (dedup)"),
    ("Acc_C_base",   "Task C full-variant accuracy (dedup by group_id)"),
    ("Acc_nocue",    "Task C nocue-variant accuracy"),
    ("Acc_cf",       "Task C counterfactual-variant accuracy"),
    ("NS_frame",     "Task C framing: avg flip rate across pos/neu/neg"),
    ("NS_frame_D",   "Task D framing: avg flip rate across pos/neu/neg"),
    ("JAccneu",      "Task D full accuracy (fallback: Task C full for gpt-5.4)"),
    ("JAccneu_D",     "Task D full accuracy (gemini-2.5 only; NA for gpt-5.4)"),
    ("Acc_D_nocue",  "Task D nocue-variant accuracy"),
    ("Acc_D_cf",     "Task D cf-variant accuracy"),
    ("IDR_raw",      "IDR: raw 3-way agreement (full/nocue/cf)"),
    ("IDR_cond",     "IDR: conditional agreement (active groups only)"),
    ("G_active_pct", "Proportion of groups not all-same across 3 variants"),
    ("IDR_source",   "taskD=canonical, taskC_fallback=gpt-5.4 has no D base data"),
]


def print_table(results: List[Dict[str, Any]]):
    print()
    header = " | ".join(f"{c:>12}" for c in METRIC_COLS)
    sep    = "-+-".join("-" * 12 for _ in METRIC_COLS)
    print(header)
    print(sep)
    for r in results:
        print(" | ".join(_fmt(r.get(c)) for c in METRIC_COLS))
    print()
    print("  Column key:")
    for col, desc in KEY_LINES:
        print(f"    {col:<14} {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute MiniWorld Composite SSOT metrics")
    parser.add_argument("--csv", help="Write CSV to this path")
    args = parser.parse_args()

    all_results = []
    for model, cfg in MODEL_CONFIGS.items():
        print(f"Processing {model}...", end=" ", flush=True)
        r = compute_model(model, cfg)
        all_results.append(r)
        missing = [k for k, v in cfg.items() if v is None or not v.exists()]
        if missing:
            print(f"partial ({len(missing)} file(s) missing: {missing})")
        else:
            print("OK")

    print_table(all_results)

    if args.csv:
        rows_out = [{c: r.get(c) for c in METRIC_COLS_RAW} for r in all_results]
        pd.DataFrame(rows_out, columns=METRIC_COLS_RAW).to_csv(args.csv, index=False)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()

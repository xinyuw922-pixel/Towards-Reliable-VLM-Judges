#!/usr/bin/env python3
"""
Build a focused audit subset for blockedunlockpickup in Task E v1.3.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROLE_TARGETS = {
    "identity_short": 3,
    "identity_medium": 3,
    "identity_long": 3,
    "status_early_holding": 3,
    "status_mid_holding": 3,
    "status_first_post_drop": 3,
    "status_late_post_drop": 3,
}


def choose_rows(cands: List[Dict[str, Any]], target: int, used_ids: set[str], used_groups: set[str]) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    ranked = sorted(
        cands,
        key=lambda r: (
            r["group_id"] in used_groups,
            r["group_id"],
            r["exam_id"],
        ),
    )
    for row in ranked:
        if row["exam_id"] in used_ids:
            continue
        chosen.append(row)
        used_ids.add(row["exam_id"])
        used_groups.add(row["group_id"])
        if len(chosen) >= target:
            break
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    exam_path = args.exam.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in exam_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    blocked_rows = [r for r in rows if r.get("env_task") == "blockedunlockpickup"]

    by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in blocked_rows:
        by_role[row["sampling_role"]].append(row)

    used_ids: set[str] = set()
    used_groups: set[str] = set()
    selected: List[Dict[str, Any]] = []
    selection_log: List[str] = []

    for role, target in ROLE_TARGETS.items():
        chosen = choose_rows(by_role.get(role, []), target, used_ids, used_groups)
        selected.extend(chosen)
        for row in chosen:
            selection_log.append(f"{role}:{row['group_id']}")
        if len(chosen) < target:
            selection_log.append(f"missing:{role}:{target-len(chosen)}")

    subset_exam = out_dir / "task_e_exam_v12.jsonl"
    subset_review = out_dir / "task_e_v13_blockedunlockpickup_focused_review.jsonl"

    with subset_exam.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_rows = []
    for row in selected:
        review_rows.append(
            {
                "exam_id": row["exam_id"],
                "image_abs": str((exam_path.parent / row["image"]).resolve()),
                "prompt": row["prompt"],
                "gold_answer": row["answer"],
                "sampling_role": row["sampling_role"],
                "env_task": row["env_task"],
                "horizon_bucket": row["horizon_bucket"],
                "query_type": row["query_type"],
                "group_id": row["group_id"],
            }
        )
    with subset_review.open("w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    link_path = images_dir / "taskE_v12"
    target_path = exam_path.parent / "images" / "taskE_v12"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    os.symlink(target_path, link_path)

    summary = {
        "source_exam": str(exam_path),
        "focus_env": "blockedunlockpickup",
        "actual_total": len(selected),
        "target_per_role": ROLE_TARGETS,
        "by_role": defaultdict(int),
        "by_query_type": defaultdict(int),
        "by_horizon_bucket": defaultdict(int),
        "selection_log": selection_log,
        "notes": [
            "This focused audit covers all available blockedunlockpickup Task E roles.",
            "blockedunlockpickup has no dedicated status-medium cell in the current v1.3 contract.",
            "status coverage is short-heavy (early_holding, mid_holding, first_post_drop) plus long late_post_drop.",
        ],
        "files": {
            "subset_exam": str(subset_exam),
            "subset_review": str(subset_review),
        },
    }
    for row in selected:
        summary["by_role"][row["sampling_role"]] += 1
        summary["by_query_type"][row["query_type"]] += 1
        summary["by_horizon_bucket"][row["horizon_bucket"]] += 1
    summary["by_role"] = dict(summary["by_role"])
    summary["by_query_type"] = dict(summary["by_query_type"])
    summary["by_horizon_bucket"] = dict(summary["by_horizon_bucket"])

    summary_path = out_dir / "focused_audit_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

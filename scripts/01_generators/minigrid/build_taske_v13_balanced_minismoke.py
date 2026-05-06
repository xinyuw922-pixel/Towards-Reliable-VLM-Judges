#!/usr/bin/env python3
"""
Build a balanced mini-smoke subset for Task E v1.3 candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROLE_TARGETS = {
    "identity_short": 5,
    "identity_medium": 5,
    "identity_long": 5,
    "status_early_holding": 5,
    "status_mid_holding": 5,
    "status_first_post_drop": 5,
    "status_late_post_drop": 5,
}

EXTRA_ROLE_PRIORITY = [
    "identity_long",
    "status_late_post_drop",
    "status_mid_holding",
    "identity_medium",
    "status_first_post_drop",
]

TARGET_PER_ENV = 36


def choose_one(cands: List[Dict[str, Any]], used_ids: set[str], used_groups: set[str]) -> Dict[str, Any] | None:
    ranked = sorted(
        cands,
        key=lambda r: (
            r["group_id"] in used_groups,
            r.get("horizon_bucket") != "long",
            r["group_id"],
            r["exam_id"],
        ),
    )
    for row in ranked:
        if row["exam_id"] not in used_ids:
            return row
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    exam_path = args.exam.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in exam_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_env_role: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_env_role[row["env_task"]][row["sampling_role"]].append(row)

    selected: List[Dict[str, Any]] = []
    selection_log: Dict[str, List[str]] = defaultdict(list)

    for env in sorted(by_env_role.keys()):
        env_selected: List[Dict[str, Any]] = []
        used_ids: set[str] = set()
        used_groups: set[str] = set()

        for role, target in ROLE_TARGETS.items():
            for idx in range(target):
                chosen = choose_one(by_env_role[env].get(role, []), used_ids, used_groups)
                if chosen is None:
                    selection_log[env].append(f"missing:{role}:{idx}")
                    continue
                env_selected.append(chosen)
                used_ids.add(chosen["exam_id"])
                used_groups.add(chosen["group_id"])
                selection_log[env].append(f"base:{role}:{chosen['group_id']}")

        for role in EXTRA_ROLE_PRIORITY:
            if len(env_selected) >= TARGET_PER_ENV:
                break
            chosen = choose_one(by_env_role[env].get(role, []), used_ids, used_groups)
            if chosen is None:
                selection_log[env].append(f"missing_extra:{role}")
                continue
            env_selected.append(chosen)
            used_ids.add(chosen["exam_id"])
            used_groups.add(chosen["group_id"])
            selection_log[env].append(f"extra:{role}:{chosen['group_id']}")

        if len(env_selected) < TARGET_PER_ENV:
            flat: List[Dict[str, Any]] = []
            for role_rows in by_env_role[env].values():
                flat.extend(role_rows)
            for row in sorted(flat, key=lambda r: (r["group_id"], r["exam_id"])):
                if len(env_selected) >= TARGET_PER_ENV:
                    break
                if row["exam_id"] in used_ids:
                    continue
                env_selected.append(row)
                used_ids.add(row["exam_id"])
                selection_log[env].append(f"fallback:{row['sampling_role']}:{row['group_id']}")

        selected.extend(env_selected)

    subset_exam = out_dir / "task_e_exam_v12.jsonl"
    subset_review = out_dir / "task_e_v13_balanced_minismoke_review.jsonl"

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
        "target_total": 72,
        "actual_total": len(selected),
        "target_per_env": TARGET_PER_ENV,
        "by_env": defaultdict(int),
        "by_role": defaultdict(int),
        "by_horizon_bucket": defaultdict(int),
        "selection_log": selection_log,
        "files": {
            "subset_exam": str(subset_exam),
            "subset_review": str(subset_review),
        },
    }
    for row in selected:
        summary["by_env"][row["env_task"]] += 1
        summary["by_role"][row["sampling_role"]] += 1
        summary["by_horizon_bucket"][row["horizon_bucket"]] += 1
    summary["by_env"] = dict(summary["by_env"])
    summary["by_role"] = dict(summary["by_role"])
    summary["by_horizon_bucket"] = dict(summary["by_horizon_bucket"])

    summary_path = out_dir / "balanced_minismoke_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

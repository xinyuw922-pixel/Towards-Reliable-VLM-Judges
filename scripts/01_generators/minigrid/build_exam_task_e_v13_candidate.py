#!/usr/bin/env python3
"""
Build a Task E v1.3 candidate exam from Phase 58 promoted environments.

Current source environments:
  - unlockpickup
  - blockedunlockpickup
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
elif str(_scripts_dir) not in _sys.path:
    _sys.path.insert(0, str(_scripts_dir))

from build_exam_task_e_v12 import (
    normalize_trace,
    extract_events,
    classify_status,
    assign_horizon_bucket,
    pick_identity_steps,
    pick_status_steps,
    select_prefix_frames,
    build_v12_row,
)


PROMOTED_ENVS = ("unlockpickup", "blockedunlockpickup")


def build_rows_for_env(
    env_name: str,
    pilot_rows: List[Dict[str, Any]],
    source_root: Path,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)

    for source in pilot_rows:
        gid = source["group_id"]
        frames = source["frames"]
        trace = normalize_trace(source["state_seq"])
        events = extract_events(trace)
        pickups = [(s, l) for s, l in events["pickups"] if l != "none"]
        drops = list(events["drops"])
        n_steps = events["n_steps"]

        if not pickups:
            stats["no_pickup"] += 1
            continue

        first_pickup_step, first_label = pickups[0]
        first_drop_step = next((s for s, l in drops if l == first_label), None)

        identity_steps = pick_identity_steps(first_pickup_step, n_steps)
        for role, qs in identity_steps.items():
            if qs is None:
                stats[f"no_candidate_{role}"] += 1
                continue
            gap = qs - first_pickup_step
            bucket = assign_horizon_bucket(gap)
            selected = select_prefix_frames(qs, first_pickup_step)
            if len(selected) > 12 or qs >= n_steps:
                stats[f"reject_{role}"] += 1
                continue
            row = build_v12_row(
                env=env_name,
                gid=gid,
                query_type="identity_of_first_picked_object",
                query_step=qs,
                anchor_step=first_pickup_step,
                horizon_gap=gap,
                horizon_bucket=bucket,
                answer=first_label,
                first_label=first_label,
                frames=frames,
                env_dir=source_root,
                out_dir=out_dir,
                row_idx=len(rows),
                sampling_role=role,
            )
            row["source_seed"] = source["seed"]
            row["source_phase"] = "58"
            rows.append(row)
            stats[f"row_{role}"] += 1

        status_steps = pick_status_steps(first_pickup_step, first_drop_step, n_steps)
        used_status_qs = set()
        for role, qs in status_steps.items():
            if qs is None:
                stats[f"no_candidate_{role}"] += 1
                continue
            if qs in used_status_qs:
                stats[f"dup_status_qs_{role}"] += 1
                continue
            gap = qs - first_pickup_step
            bucket = assign_horizon_bucket(gap)
            status_answer = classify_status(first_label, pickups, drops, events["reacquires"], qs)
            if role.startswith("status_") and "holding" in role and status_answer != "holding":
                stats[f"status_not_holding_{role}"] += 1
                continue
            selected = select_prefix_frames(qs, first_pickup_step)
            if len(selected) > 12 or qs >= n_steps:
                stats[f"reject_{role}"] += 1
                continue
            row = build_v12_row(
                env=env_name,
                gid=gid,
                query_type="status_of_first_picked_object_at_checkpoint",
                query_step=qs,
                anchor_step=first_pickup_step,
                horizon_gap=gap,
                horizon_bucket=bucket,
                answer=status_answer,
                first_label=first_label,
                frames=frames,
                env_dir=source_root,
                out_dir=out_dir,
                row_idx=len(rows),
                sampling_role=role,
            )
            row["source_seed"] = source["seed"]
            row["source_phase"] = "58"
            rows.append(row)
            used_status_qs.add(qs)
            stats[f"row_{role}"] += 1

    return rows, dict(stats)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_env = Counter()
    by_role = Counter()
    by_query_type = Counter()
    by_horizon = Counter()
    by_env_role = defaultdict(Counter)
    for row in rows:
        by_env[row["env_task"]] += 1
        by_role[row["sampling_role"]] += 1
        by_query_type[row["query_type"]] += 1
        by_horizon[row["horizon_bucket"]] += 1
        by_env_role[row["env_task"]][row["sampling_role"]] += 1
    return {
        "n_rows": len(rows),
        "by_env": dict(by_env),
        "by_sampling_role": dict(by_role),
        "by_query_type": dict(by_query_type),
        "by_horizon_bucket": dict(by_horizon),
        "by_env_role": {k: dict(v) for k, v in by_env_role.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase58-root",
        type=Path,
        default=Path("runs/phase58_taske_env_screen_2026-03-31"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/exams_taske_v13_candidate"),
    )
    args = ap.parse_args()

    source_root = args.phase58_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    build_stats: Dict[str, Any] = {}

    for env_name in PROMOTED_ENVS:
        pilot_path = source_root / f"{env_name}_pilot_full.jsonl"
        pilot_rows = [
            json.loads(line)
            for line in pilot_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows, stats = build_rows_for_env(env_name, pilot_rows, source_root, out_dir)
        build_stats[env_name] = stats
        all_rows.extend(rows)

    exam_path = out_dir / "task_e_exam_v13.jsonl"
    with exam_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_rows = []
    for row in all_rows:
        review_rows.append(
            {
                "exam_id": row["exam_id"],
                "image_abs": str((out_dir / row["image"]).resolve()),
                "prompt": row["prompt"],
                "gold_answer": row["answer"],
                "sampling_role": row["sampling_role"],
                "env_task": row["env_task"],
                "horizon_bucket": row["horizon_bucket"],
                "query_type": row["query_type"],
            }
        )
    review_path = out_dir / "human_audit_package_v13.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "task": "E",
        "version": "v1.3-candidate",
        "source_phase": "58",
        "promoted_envs": list(PROMOTED_ENVS),
        "summary": summarize(all_rows),
        "build_stats": build_stats,
        "files": {
            "exam": str(exam_path),
            "human_audit_package": str(review_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build an alternate Task D artifact under the Memory redesign:

- For Memory rows only, drop `nocue`
- Keep `full` and `cf`
- Prepend a real cue-room first frame to those Memory rows
- Leave all non-Memory rows unchanged

This does not overwrite the default Task D authority; it creates a reviewable
alternate exam so we can inspect the redesign before promotion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_root = _scripts_dir.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root.parent))

from scripts03_postprocess.audit.apply_memory_hybrid_to_taskd import build_memory_hybrid_montage, load_group_records


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Task D alternate artifact: Memory full+cf only")
    ap.add_argument(
        "--source-exam",
        type=Path,
        default=Path("outputs/exams_taskd/task_d_exam.jsonl"),
    )
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=Path("outputs/raw_data"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/exams_taskd"),
    )
    args = ap.parse_args()

    rows = read_jsonl(args.source_exam)
    memory_groups = load_group_records(args.raw_root, "memory")

    out_rows: List[Dict[str, Any]] = []
    changed_memory_rows: List[Dict[str, Any]] = []
    removed_memory_nocue: List[str] = []
    img_root = args.out_dir / "images" / "taskD_memory_full_cf_redesign"

    for row in rows:
        if row.get("env_task") != "memory":
            out_rows.append(row)
            continue

        variant = row.get("source_variant") or row.get("variant")
        gid = row.get("source_group_id") or row.get("group_id")
        if variant == "nocue":
            removed_memory_nocue.append(row["uid"])
            continue

        rec = memory_groups[gid][variant]
        montage, extra = build_memory_hybrid_montage(args.raw_root, "memory", rec)
        rel = f"memory/{gid}/{variant}_orig_clean_hybrid.png"
        full_path = img_root / rel
        full_path.parent.mkdir(parents=True, exist_ok=True)
        montage.save(full_path, "PNG")

        new_row = dict(row)
        new_row["image"] = f"images/taskD_memory_full_cf_redesign/{rel}"
        new_row["n_frames"] = int(extra["meta"]["n_frames_hybrid"])
        new_row["layout"] = extra["layout"]
        new_row["hybrid_memory_first_frame"] = True
        new_row["memory_cue_room_bbox"] = extra["meta"]["cue_room_bbox"]
        new_row["memory_redesign_mode"] = "full_cf_only_with_real_cue_first_frame"
        out_rows.append(new_row)
        changed_memory_rows.append(
            {
                "uid": new_row["uid"],
                "group_id": gid,
                "variant": variant,
                "image": new_row["image"],
                "cue_room_bbox": extra["meta"]["cue_room_bbox"],
            }
        )

    exam_path = args.out_dir / "task_d_exam_memory_full_cf_redesign.jsonl"
    write_jsonl(exam_path, out_rows)

    env_counts = Counter(row["env_task"] for row in out_rows)
    variant_counts = Counter(row["variant"] for row in out_rows)
    answer_counts = Counter(row["answer"] for row in out_rows)

    manifest = {
        "source_exam": str(args.source_exam),
        "output_exam": str(exam_path),
        "mode": "taskd_memory_full_cf_only_redesign_v1",
        "summary": {
            "rows_total": len(out_rows),
            "memory_rows_changed": len(changed_memory_rows),
            "memory_nocue_rows_removed": len(removed_memory_nocue),
        },
        "counts": {
            "env_counts": dict(env_counts),
            "variant_counts": dict(variant_counts),
            "answer_counts": dict(answer_counts),
        },
        "removed_memory_nocue_uids": removed_memory_nocue,
        "changed_memory_rows": changed_memory_rows,
    }
    write_json(args.out_dir / "memory_full_cf_redesign_manifest.json", manifest)

    print(
        json.dumps(
            {
                "output_exam": str(exam_path),
                "rows_total": len(out_rows),
                "memory_rows_changed": len(changed_memory_rows),
                "memory_nocue_rows_removed": len(removed_memory_nocue),
                "variant_counts": dict(variant_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

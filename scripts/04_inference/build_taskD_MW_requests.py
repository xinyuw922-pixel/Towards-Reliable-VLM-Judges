#!/usr/bin/env python3
"""
Build run_inference requests for D-MW Trajectory Judgment.

Usage:
  python3 build_taskD_MW_requests.py \
    --exam_jsonl tmp_miniworld/taskA-MW-D-fourrooms/taskA-MW-D-fourrooms.jsonl \
    --out tmp_miniworld/taskA-MW-D-fourrooms/requests_D_MW.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(record):
    base_prompt = str(record.get("prompt", "")).strip()
    base_prompt = re.sub(
        r"\n\nAnswer with exactly one uppercase letter: A or B\.\nDo not output any other words\.\s*$",
        "",
        base_prompt,
        flags=re.DOTALL,
    ).strip()

    order_note = (
        "The input consists of three images shown in this exact order:\n"
        "1. the initial state image\n"
        "2. candidate outcome A\n"
        "3. candidate outcome B\n\n"
        "The initial state image shows the agent's starting position and direction.\n"
        "Each candidate image shows the agent's final state after the planned trajectory.\n"
    )
    output_contract = (
        "\n\nOutput rule:\n"
        "Your entire response must be exactly one uppercase letter: A or B.\n"
        "Do not output any other character, word, explanation, punctuation, or newline.\n"
        "If you think the answer is candidate A, output exactly: A\n"
        "If you think the answer is candidate B, output exactly: B"
    )
    return f"{order_note}{base_prompt}{output_contract}"


def main():
    ap = argparse.ArgumentParser(description="Build D-MW requests")
    ap.add_argument(
        "--exam_jsonl",
        default="tmp_miniworld/taskA-MW-D-fourrooms/taskA-MW-D-fourrooms.jsonl",
        help="Path to D-MW exam JSONL",
    )
    ap.add_argument(
        "--out",
        default="tmp_miniworld/taskA-MW-D-fourrooms/requests_D_MW.jsonl",
        help="Output requests JSONL path",
    )
    args = ap.parse_args()

    exam_jsonl = Path(args.exam_jsonl)
    out_path = Path(args.out)
    exam_root = exam_jsonl.parent
    rows = load_jsonl(exam_jsonl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            initial_path = str((exam_root / row["initial_image_rel_path"]).resolve())
            candidate_a_path = str((exam_root / row["candidate_a_rel_path"]).resolve())
            candidate_b_path = str((exam_root / row["candidate_b_rel_path"]).resolve())

            payload = {
                "uid": row["uid"],
                "exam_id": row["exam_id"],
                "schema_version": row.get("schema_version", "gridwm.mw_exam.v2"),
                "images": [initial_path, candidate_a_path, candidate_b_path],
                "prompt": build_prompt(row),
                "exam": {
                    "uid": row["uid"],
                    "exam_id": row["exam_id"],
                    "task": row.get("task", "D-MW"),
                    "env_task": row.get("env_task"),
                    "subtask": row.get("subtask"),
                    "schema_version": row.get("schema_version", "gridwm.mw_exam.v2"),
                    "correct_candidate": row.get("correct_candidate"),
                    "label": row.get("label"),
                    "distractor_strategy": row.get("distractor_strategy"),
                    "planned_trajectory": row.get("planned_trajectory"),
                    "success_dir": row.get("success_dir"),
                    "failure_dir": row.get("failure_dir"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Built {len(rows)} requests -> {out_path}")


if __name__ == "__main__":
    main()

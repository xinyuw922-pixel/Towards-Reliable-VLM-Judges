#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(record: dict[str, Any]) -> str:
    base_prompt = str(record.get("prompt", "")).strip()
    # Normalize any previously appended request-level answer contract so the
    # builder can be run repeatedly without duplicating or preserving old tails.
    base_prompt = re.sub(
        r"\n\nAnswer with exactly one uppercase letter: A or B\.\nDo not output any other words\.\s*$",
        "",
        base_prompt,
        flags=re.DOTALL,
    ).strip()
    order_note = (
        "The input consists of three images shown in this exact order:\n"
        "1. the current state image\n"
        "2. candidate view A\n"
        "3. candidate view B\n\n"
        "The current state image contains the agent's ego view together with a top-view map.\n"
        "Each candidate image shows only the candidate successor ego view.\n"
    )
    # Keep a hard single-letter output contract here by default.
    # Gemini improved when the prompt explicitly forbade any extra tokens.
    output_contract = (
        "\n\nOutput rule:\n"
        "Your entire response must be exactly one uppercase letter: A or B.\n"
        "Do not output any other character, word, explanation, punctuation, or newline.\n"
        "If you think the answer is candidate A, output exactly: A\n"
        "If you think the answer is candidate B, output exactly: B"
    )
    return f"{order_note}{base_prompt}{output_contract}"


def main() -> None:
    # Get repo root dynamically
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent.parent
    
    ap = argparse.ArgumentParser(
        description="Build run_inference requests for Task A-MW v2 with strict single-letter output contract"
    )
    ap.add_argument(
        "--exam_jsonl",
        default=str(repo_root / "tmp_miniworld" / "taskA-MW-v2-preflight" / "taskA-MW-v2-prototype.jsonl"),
        help="Path to Task A-MW v2 exam JSONL",
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "tmp_miniworld" / "taskA-MW-v2-preflight" / "requests_A_MW_v2.jsonl"),
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
            current_path = str((exam_root / row["current_image_rel_path"]).resolve())
            candidate_a_path = str((exam_root / row["candidate_a_rel_path"]).resolve())
            candidate_b_path = str((exam_root / row["candidate_b_rel_path"]).resolve())
            payload = {
                "uid": row["uid"],
                "exam_id": row["exam_id"],
                "schema_version": row.get("schema_version", "gridwm.mw_exam.v2"),
                "images": [current_path, candidate_a_path, candidate_b_path],
                "prompt": build_prompt(row),
                "exam": {
                    "uid": row["uid"],
                    "exam_id": row["exam_id"],
                    "task": row.get("task", "A-MW-v2"),
                    "env_task": row.get("env_task"),
                    "subtask": row.get("subtask"),
                    "schema_version": row.get("schema_version", "gridwm.mw_exam.v2"),
                    "difficulty_bucket": row.get("difficulty_bucket"),
                    "correct_candidate": row.get("correct_candidate"),
                    "label": row.get("label"),
                    "distractor_strategy": row.get("distractor_strategy"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Built {len(rows)} requests -> {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Below are two time-ordered MiniWorld trajectory storyboards from the same task family.\n\n"
    "Reference trajectory:\n"
    "This is a known completed example of the task family.\n\n"
    "Query trajectory:\n"
    "Determine whether the query trajectory also completes the same task goal as the reference example.\n"
    "The room layout and path complexity may differ, but the task rule is the same.\n\n"
    "Answer with ONLY: Yes or No."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build requests JSONL for frozen MiniWorld reference-family pilot")
    ap.add_argument(
        "--exam_jsonl",
        default="tmp_miniworld/taskD-MW-reference-family-pilot-freeze/task_d_exam.jsonl",
    )
    ap.add_argument(
        "--out",
        default="tmp_miniworld/taskD-MW-reference-family-pilot-freeze/requests_D_MW_reference_family_pilot.jsonl",
    )
    args = ap.parse_args()

    exam_jsonl = Path(args.exam_jsonl)
    out_path = Path(args.out)
    exam_root = exam_jsonl.parent
    rows = load_jsonl(exam_jsonl)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            uid = str(row.get("uid") or row.get("exam_id"))
            image_path = str((exam_root / row["image"]).resolve())
            payload = {
                "uid": uid,
                "exam_id": str(row.get("exam_id", uid)),
                "schema_version": str(row.get("schema_version", "gridwm.mw_exam.rf.v1")),
                "images": [image_path],
                # Use a D7-safe prompt for online inference even if the frozen exam row
                # still carries the earlier Success/Fail wording.
                "prompt": DEFAULT_PROMPT,
                "exam": {
                    "uid": uid,
                    "exam_id": str(row.get("exam_id", uid)),
                    "task": "D",
                    "subtask": "D-MW-reference-family-pilot",
                    "schema_version": str(row.get("schema_version", "gridwm.mw_exam.rf.v1")),
                    "family": row.get("family"),
                    "reference_template_id": row.get("reference_template_id"),
                    "reference_difficulty": row.get("reference_difficulty"),
                    "query_template_id": row.get("query_template_id"),
                    "query_difficulty": row.get("query_difficulty"),
                    "query_variant": row.get("query_variant"),
                    "answer": row.get("answer"),
                    "redblue_failure_rule": row.get("redblue_failure_rule"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "num_requests": len(rows),
        "exam_jsonl": str(exam_jsonl),
        "requests_jsonl": str(out_path),
        "task": "D",
        "subtask": "D-MW-reference-family-pilot",
    }
    (out_path.parent / "requests_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built {len(rows)} requests -> {out_path}")


if __name__ == "__main__":
    main()

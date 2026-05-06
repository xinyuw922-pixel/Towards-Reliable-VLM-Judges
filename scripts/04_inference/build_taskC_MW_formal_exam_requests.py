#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    # Get repo root dynamically
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent.parent
    
    ap = argparse.ArgumentParser(description="Build requests JSONL for the formal Task C-MW exam bank")
    ap.add_argument(
        "--exam_jsonl",
        default=str(repo_root / "tmp_miniworld" / "taskC-MW-formal-bank" / "task_c_exam.jsonl"),
    )
    ap.add_argument(
        "--out",
        default=str(repo_root / "tmp_miniworld" / "taskC-MW-formal-bank" / "requests_C_MW_formal_exam.jsonl"),
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
            image_root = Path(str(row.get("image_root", exam_root)))
            image_path = str((image_root / row["image"]).resolve())
            payload = {
                "uid": uid,
                "exam_id": str(row.get("exam_id", uid)),
                "schema_version": str(row.get("schema_version", "gridwm.mw_exam.v1")),
                "images": [image_path],
                "prompt": str(row["prompt"]),
                "exam": {
                    "uid": uid,
                    "exam_id": str(row.get("exam_id", uid)),
                    "task": "C",
                    "subtask": "C-MW-formal",
                    "schema_version": str(row.get("schema_version", "gridwm.mw_exam.v1")),
                    "family": row.get("family"),
                    "template_id": row.get("template_id"),
                    "variant": row.get("variant"),
                    "difficulty": row.get("difficulty"),
                    "answer": row.get("answer"),
                    "task_goal": row.get("task_goal"),
                    "source_summary": row.get("source_summary"),
                    "selected_steps": row.get("selected_steps"),
                    "keyframe_selector": row.get("keyframe_selector"),
                    "nocue_meta": row.get("nocue_meta"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    manifest = {
        "num_requests": len(rows),
        "exam_jsonl": str(exam_jsonl),
        "requests_jsonl": str(out_path),
        "task": "C",
        "subtask": "C-MW-formal",
        "prompt_source": "exam_row.prompt",
    }
    (out_path.parent / "requests_build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built {len(rows)} requests -> {out_path}")


if __name__ == "__main__":
    main()

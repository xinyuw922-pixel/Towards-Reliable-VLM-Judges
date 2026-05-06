#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


FRAMING_PREAMBLES = {
    "pos": "The query trajectory seems likely to satisfy the same task as the reference example.",
    "neu": "Below are two time-ordered MiniWorld trajectory storyboards from the same task family.",
    "neg": "The query trajectory seems likely to fail the same task as the reference example.",
}

DEFAULT_PROMPT_BODY = (
    "Reference trajectory:\n"
    "This is a known completed example of the task family.\n\n"
    "Query trajectory:\n"
    "Determine whether the query trajectory also completes the same task goal as the reference example.\n"
    "The room layout and path complexity may differ, but the task rule is the same.\n\n"
    "Answer with ONLY: Yes or No."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(first_line: str, row: dict[str, Any]) -> str:
    base = str(row.get("prompt") or "").strip()
    if base:
        lines = base.splitlines()
        lines[0] = first_line
        return "\n".join(lines)
    return f"{first_line}\n\n{DEFAULT_PROMPT_BODY}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build full Task D-MW framing exam with neu/pos/neg prompt variants")
    ap.add_argument(
        "--exam_jsonl",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/task_d_exam.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/framing_full",
    )
    args = ap.parse_args()

    exam_rows = load_jsonl(Path(args.exam_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.exam_jsonl).resolve().parent

    out_rows: list[dict[str, Any]] = []
    for row in exam_rows:
        src_img = source_root / str(row["image"])
        dst_img = out_dir / str(row["image"])
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        for framing, preamble in FRAMING_PREAMBLES.items():
            new_row = dict(row)
            uid_base = str(row.get("uid") or row.get("exam_id"))
            new_uid = f"{uid_base}.{framing}"
            new_row["uid"] = new_uid
            new_row["exam_id"] = new_uid
            new_row["framing"] = framing
            new_row["prompt"] = build_prompt(preamble, row)
            out_rows.append(new_row)

    exam_out = out_dir / "task_d_exam_framing_full.jsonl"
    with exam_out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "base_exam_jsonl": args.exam_jsonl,
        "exam_jsonl": str(exam_out),
        "num_base_rows": len(exam_rows),
        "num_framing_rows": len(out_rows),
        "framing_preambles": FRAMING_PREAMBLES,
        "design": "same image, same gold label, same prompt body; only first sentence varies",
    }
    (out_dir / "framing_full_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built {len(out_rows)} framing rows -> {exam_out}")


if __name__ == "__main__":
    main()

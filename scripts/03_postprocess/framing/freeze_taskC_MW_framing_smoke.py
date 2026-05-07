#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze Task C-MW framing smoke pack")
    ap.add_argument(
        "--src_dir",
        default="tmp_miniworld/taskC-MW-formal-bank/framing_smoke",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/taskC-MW-formal-bank/framing_smoke_freeze",
    )
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exam_path = src_dir / "task_c_exam_framing_smoke.jsonl"
    requests_path = src_dir / "requests_C_MW_framing_smoke_v3.jsonl"
    framing_manifest = src_dir / "framing_smoke_manifest.json"
    audit_note = src_dir / "gpt54_framing_smoke_note.md"
    response_jsonl = (
        src_dir
        / "responses_gpt54_responses_v3"
        / "openaicompatible_zhizengzeng_gpt-5.4_responses"
        / "requests_C_MW_framing_smoke_v3.jsonl"
    )
    response_per_row = (
        src_dir
        / "responses_gpt54_responses_v3"
        / "openaicompatible_zhizengzeng_gpt-5.4_responses"
        / "score_per_row.jsonl"
    )
    response_summary = src_dir / "responses_gpt54_responses_v3" / "smoke_summary.json"

    rows = load_jsonl(exam_path)
    copied_images: set[str] = set()
    for row in rows:
        rel = str(row["image"])
        if rel in copied_images:
            continue
        copied_images.add(rel)
        src_img = src_dir / rel
        dst_img = out_dir / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

    shutil.copy2(exam_path, out_dir / exam_path.name)
    shutil.copy2(requests_path, out_dir / "requests_C_MW_framing_smoke.jsonl")
    shutil.copy2(framing_manifest, out_dir / framing_manifest.name)
    shutil.copy2(audit_note, out_dir / audit_note.name)

    freeze_resp_dir = out_dir / "responses_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses"
    freeze_resp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(response_jsonl, freeze_resp_dir / "requests_C_MW_framing_smoke.jsonl")
    shutil.copy2(response_per_row, freeze_resp_dir / response_per_row.name)
    shutil.copy2(response_summary, out_dir / "responses_gpt54_responses" / response_summary.name)

    summary_obj = json.loads(response_summary.read_text(encoding="utf-8"))
    frozen_summary = {
        "task": "C-MW-framing-smoke",
        "status": "frozen",
        "num_records": len(rows),
        "num_unique_images": len(copied_images),
        "source_dir": str(src_dir),
        "valid_run": "v3",
        "smoke_model": "zhizengzeng / gpt-5.4 / responses",
        "accuracy": summary_obj.get("accuracy"),
        "by_framing": summary_obj.get("by_framing", {}),
        "by_variant": summary_obj.get("by_variant", {}),
        "smoke_summary_relpath": "responses_gpt54_responses/smoke_summary.json",
    }
    write_json(out_dir / "taskC_MW_framing_smoke_freeze_summary.json", frozen_summary)

    note = "\n".join(
        [
            "# Task C-MW Framing Smoke Freeze",
            "",
            "Status: Frozen",
            "",
            "Contents:",
            "- Framing smoke exam JSONL (54 rows)",
            "- Valid requests JSONL (v3)",
            "- gpt-5.4 + responses valid responses and summary (v3)",
            "- Framing error audit note",
            "- Related image assets",
            "",
            "Key Conclusions:",
            f"- pos: {summary_obj.get('by_framing', {}).get('pos', {}).get('correct', 0)}/{summary_obj.get('by_framing', {}).get('pos', {}).get('n', 0)}",
            f"- neu: {summary_obj.get('by_framing', {}).get('neu', {}).get('correct', 0)}/{summary_obj.get('by_framing', {}).get('neu', {}).get('n', 0)}",
            f"- neg: {summary_obj.get('by_framing', {}).get('neg', {}).get('correct', 0)}/{summary_obj.get('by_framing', {}).get('neg', {}).get('n', 0)}",
            "",
            "Notes:",
            "- v1 / v2 runs had image path issues, excluded from freeze pack",
            "- This freeze pack only includes v3 valid results",
            "",
            f"Source directory: {src_dir}",
        ]
    )
    (out_dir / "freeze_note.md").write_text(note, encoding="utf-8")
    print(f"Saved Task C-MW framing smoke freeze pack to {out_dir}")


if __name__ == "__main__":
    main()

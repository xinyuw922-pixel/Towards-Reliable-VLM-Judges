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
    ap = argparse.ArgumentParser(description="Freeze Task D-MW reference-family exam50 pack")
    ap.add_argument(
        "--src_dir",
        default="tmp_miniworld/taskD-MW-reference-family-exam50",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/taskD-MW-reference-family-exam50-freeze",
    )
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exam_path = src_dir / "task_d_mw_reference_family_exam50.jsonl"
    task_d_exam = src_dir / "task_d_exam.jsonl"
    review_pdf = src_dir / "task_d_mw_reference_family_exam50_review_zh.pdf"
    summary_json = src_dir / "reference_family_exam50_summary.json"
    prompt_txt = src_dir / "task_d_mw_reference_family_prompt.txt"
    requests_jsonl = src_dir / "requests_D_MW_reference_family_exam50.jsonl"
    requests_manifest = src_dir / "requests_build_manifest.json"
    response_jsonl = src_dir / "responses_full_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses" / "requests_D_MW_reference_family_exam50.jsonl"
    response_summary = src_dir / "responses_full_gpt54_responses" / "smoke_summary.json"

    rows = load_jsonl(exam_path)
    for row in rows:
        src_img = src_dir / row["image"]
        dst_img = out_dir / row["image"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

    for src in (
        exam_path,
        task_d_exam,
        review_pdf,
        summary_json,
        prompt_txt,
        requests_jsonl,
        requests_manifest,
    ):
        shutil.copy2(src, out_dir / src.name)

    freeze_resp_dir = out_dir / "responses_full_gpt54_responses" / "openaicompatible_zhizengzeng_gpt-5.4_responses"
    freeze_resp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(response_jsonl, freeze_resp_dir / response_jsonl.name)
    shutil.copy2(response_summary, out_dir / "responses_full_gpt54_responses" / response_summary.name)

    source_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    frozen_summary = {
        "task": "D-MW-reference-family-exam50",
        "status": "frozen",
        "num_records": source_summary["num_records"],
        "num_query_templates": source_summary["num_query_templates"],
        "families": source_summary["families"],
        "source_buckets": source_summary["source_buckets"],
        "source_dir": str(src_dir),
        "smoke_model": "zhizengzeng / gpt-5.4 / responses",
        "smoke_summary_relpath": "responses_full_gpt54_responses/smoke_summary.json",
    }
    write_json(out_dir / "taskD_MW_reference_family_exam50_freeze_summary.json", frozen_summary)

    note = "\n".join(
        [
            "# Task D-MW Reference-Family Exam50 Freeze",
            "",
            "状态：冻结完成",
            "",
            "冻结内容：",
            "- 50 题 reference-family exam JSONL",
            "- 中文审阅 PDF",
            "- requests JSONL",
            "- gpt-5.4 + responses 全量 smoke 响应与摘要",
            "",
            "本轮结论：",
            "- 题量：50",
            "- query templates：25",
            "- smoke model：zhizengzeng / gpt-5.4 / responses",
            "- summary：responses_full_gpt54_responses/smoke_summary.json",
            "",
            "冻结原因：",
            "- Task D-MW 已形成稳定可运行的 Tier-1 evidence pack",
            "- 当前更值得把主精力切换到 Task C-MW 设计与原型，而不是继续扩大 Task D-MW 题量",
            "",
            f"源目录：{src_dir}",
        ]
    )
    (out_dir / "freeze_note.md").write_text(note, encoding="utf-8")
    print(f"Saved Task D-MW freeze pack to {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


KEEP_DIFFICULTIES = {"hard", "structural", "structural_v2"}
KEEP_FAMILIES = ("doorkey", "multiroom", "redblue")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a selected MiniWorld reference-family pilot pack")
    ap.add_argument(
        "--src_dir",
        default="tmp_miniworld/taskD-MW-reference-family-showcase",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/taskD-MW-reference-family-pilot-freeze",
    )
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exam_path = src_dir / "task_d_mw_reference_family_exam.jsonl"
    summary_path = src_dir / "reference_family_summary.json"
    prompt_path = src_dir / "task_d_mw_reference_family_prompt.txt"

    rows = load_jsonl(exam_path)
    selected = [
        row
        for row in rows
        if row["family"] in KEEP_FAMILIES and str(row["query_difficulty"]) in KEEP_DIFFICULTIES
    ]
    selected.sort(key=lambda r: (KEEP_FAMILIES.index(r["family"]), str(r["query_difficulty"]), r["query_variant"], r["query_template_id"]))

    for row in selected:
        src_img = src_dir / row["image"]
        dst_img = out_dir / row["image"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)

    write_jsonl(out_dir / "task_d_mw_reference_family_exam.jsonl", selected)
    write_jsonl(out_dir / "task_d_exam.jsonl", selected)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frozen_summary = {
        "selection": {
            "families": list(KEEP_FAMILIES),
            "difficulties": sorted(KEEP_DIFFICULTIES),
            "variants": ["full", "cf"],
        },
        "num_records": len(selected),
        "source_dir": str(src_dir),
        "records": selected,
        "families": {
            family: {
                "num_records": sum(1 for row in selected if row["family"] == family),
                "reference_template": source_summary["families"][family]["reference_template"],
            }
            for family in KEEP_FAMILIES
        },
    }
    write_json(out_dir / "reference_family_pilot_summary.json", frozen_summary)

    note = "\n".join(
        [
            "# Task D-MW Reference-Family Pilot Freeze",
            "",
            "冻结范围：",
            "- family: doorkey / multiroom / redblue",
            "- difficulty: hard / structural / structural_v2",
            "- variant: full / cf",
            "",
            f"题目总数：{len(selected)}",
            "",
            "冻结目的：",
            "- 先固定一版更有区分度的 pilot pack",
            "- 暂停继续扩题",
            "- 先做本地 smoke 和后续小规模模型试跑",
            "",
            "规则说明：",
            "- RedBlue 的 cf 使用 swap_redblue_order，而不是 lock_gate2_forever",
            "- 当前这条线是 reference-family pilot，不是 SSOT 主线 Task D 最终合同",
            "- 目录内额外提供 task_d_exam.jsonl 兼容别名，便于直接接入通用 Task D 推理入口",
            "",
            f"源目录：{src_dir}",
        ]
    )
    (out_dir / "pilot_freeze_note.md").write_text(note, encoding="utf-8")
    shutil.copy2(prompt_path, out_dir / prompt_path.name)
    print(f"Saved pilot freeze to {out_dir}")


if __name__ == "__main__":
    main()

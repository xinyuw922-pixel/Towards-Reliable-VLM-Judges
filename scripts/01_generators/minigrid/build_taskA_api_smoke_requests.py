#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON_DIR = ROOT / "scripts" / "tmp_taskA_image_format_comparison"
DEFAULT_CANONICAL_EXAM_DIR = ROOT / "outputs" / "exams" / "taska"

STRICT_SUFFIX = (
    "\nAnswer with ONLY the correct letter (A/B/C/D). "
    "Return exactly one uppercase letter and no other text."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_request(row: dict[str, Any], asset_root: Path, api_content_order: str = "", api_image_detail: str = "") -> dict[str, Any]:
    uid = str(row["uid"])
    if "images" in row:
        images = [
            str((asset_root / rel) if not Path(rel).is_absolute() else Path(rel))
            for rel in row["images"]
        ]
    else:
        rel = row["image"]
        images = [str((asset_root / rel) if not Path(rel).is_absolute() else Path(rel))]

    prompt = str(row.get("prompt", "")).strip()
    if STRICT_SUFFIX.strip() not in prompt:
        prompt = (prompt + STRICT_SUFFIX).strip()

    req: dict[str, Any] = {
        "uid": uid,
        "exam_id": row.get("exam_id", uid),
        "schema_version": "gridwm.exam.v1",
        "images": images,
        "prompt": prompt,
        "exam": {
            "uid": uid,
            "exam_id": row.get("exam_id", uid),
            "task": row.get("task", "A"),
            "env_task": row.get("env_task"),
            "group_id": str(uid).split(".")[2] if len(str(uid).split(".")) >= 4 else None,
            "schema_version": row.get("schema_version", "gridwm.exam.v1"),
        },
    }
    if api_content_order:
        req["api_content_order"] = api_content_order
    if api_image_detail:
        req["api_image_detail"] = api_image_detail
    return req


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Task A API smoke requests for composite vs split images.")
    ap.add_argument("--comparison_dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    ap.add_argument("--canonical_exam_dir", type=Path, default=DEFAULT_CANONICAL_EXAM_DIR)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num", type=int, default=45, help="Number of paired items to export. Use 0 or a value >= total to export all rows.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ordered", action="store_true", help="Preserve canonical exam order instead of random sampling.")
    ap.add_argument("--api_content_order", default="", help="Value for api_content_order field in each request (e.g. 'text_first'). If empty, not included.")
    ap.add_argument("--api_image_detail", default="", help="Value for api_image_detail field in each request (e.g. 'auto'). If empty, not included.")
    args = ap.parse_args()

    comparison_dir = args.comparison_dir
    orig_path = comparison_dir / "task_a_exam_original.jsonl"
    split_path = comparison_dir / "task_a_exam_split.jsonl"

    if not orig_path.exists() or not split_path.exists():
        raise FileNotFoundError(
            f"Expected both {orig_path} and {split_path}. "
            "Run scripts/split_taskA_frames.py first."
        )

    orig_rows = load_jsonl(orig_path)
    split_rows = load_jsonl(split_path)
    orig_by_uid = {r["uid"]: r for r in orig_rows}
    split_by_uid = {r["uid"]: r for r in split_rows}
    common_uids = [r["uid"] for r in orig_rows if r["uid"] in split_by_uid]
    total_common = len(common_uids)
    if total_common == 0:
        raise ValueError("No common rows found between original and split Task A exams.")

    if args.num <= 0 or args.num >= total_common:
        sampled_uids = list(common_uids)
    elif args.ordered:
        sampled_uids = common_uids[: args.num]
    else:
        rng = random.Random(args.seed)
        sampled_uids = rng.sample(common_uids, args.num)

    composite_exam_dir = args.out_dir / "composite"
    split_exam_dir = args.out_dir / "split"
    composite_exam_dir.mkdir(parents=True, exist_ok=True)
    split_exam_dir.mkdir(parents=True, exist_ok=True)

    sampled_orig = [orig_by_uid[uid] for uid in sampled_uids]
    sampled_split = [split_by_uid[uid] for uid in sampled_uids]

    write_jsonl(composite_exam_dir / "task_a_exam.jsonl", sampled_orig)
    write_jsonl(split_exam_dir / "task_a_exam.jsonl", sampled_split)

    composite_requests = [make_request(r, args.canonical_exam_dir, args.api_content_order, args.api_image_detail) for r in sampled_orig]
    split_requests = [make_request(r, comparison_dir, args.api_content_order, args.api_image_detail) for r in sampled_split]
    write_jsonl(composite_exam_dir / "requests_taskA_composite.jsonl", composite_requests)
    write_jsonl(split_exam_dir / "requests_taskA_split.jsonl", split_requests)

    manifest = {
        "comparison_dir": str(comparison_dir),
        "canonical_exam_dir": str(args.canonical_exam_dir),
        "out_dir": str(args.out_dir),
        "num": len(sampled_uids),
        "requested_num": args.num,
        "seed": args.seed,
        "ordered": args.ordered,
        "total_common": total_common,
        "uids": sampled_uids,
        "api_content_order": args.api_content_order or None,
        "api_image_detail": args.api_image_detail or None,
        "composite_exam": str((composite_exam_dir / "task_a_exam.jsonl").relative_to(args.out_dir)),
        "split_exam": str((split_exam_dir / "task_a_exam.jsonl").relative_to(args.out_dir)),
        "composite_requests": str((composite_exam_dir / "requests_taskA_composite.jsonl").relative_to(args.out_dir)),
        "split_requests": str((split_exam_dir / "requests_taskA_split.jsonl").relative_to(args.out_dir)),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

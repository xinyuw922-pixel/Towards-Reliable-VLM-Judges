#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "runs" / "phase91_taskA_split_full_matrix_20260419" / "request_bundle" / "split" / "requests_taskA_split.jsonl"


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


def clone_rows(rows: list[dict[str, Any]], **extra: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(extra)
        out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build tiny Task A split-image format probes.")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    rows = load_jsonl(args.source)
    if not rows:
        raise ValueError(f"No rows found in {args.source}")

    chosen = rows[: args.num]
    variants = {
        "baseline_images_first": clone_rows(chosen),
        "probe_text_first": clone_rows(chosen, api_content_order="text_first"),
        "probe_detail_auto": clone_rows(chosen, api_image_detail="auto"),
        "probe_text_first_detail_auto": clone_rows(
            chosen, api_content_order="text_first", api_image_detail="auto"
        ),
    }

    for name, variant_rows in variants.items():
        write_jsonl(args.out_dir / name / f"{name}.jsonl", variant_rows)

    manifest = {
        "source": str(args.source),
        "num": len(chosen),
        "uids": [r["uid"] for r in chosen],
        "variants": {
            name: str((args.out_dir / name / f"{name}.jsonl").relative_to(args.out_dir))
            for name in variants
        },
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

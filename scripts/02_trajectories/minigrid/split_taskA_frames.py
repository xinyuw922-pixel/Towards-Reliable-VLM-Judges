#!/usr/bin/env python3
"""
Task A Image Splitting Tool — Compare split-image format vs composite-image format.

Splits canonical Task A composite images (3 columns x 2 rows = 5 regions)
into 5 separate PNG images:
  - current.png   — Current state frame (vertically centered in left column)
  - choice_A.png — Candidate A (top-left with A label)
  - choice_B.png — Candidate B (top-right with B label)
  - choice_C.png — Candidate C (bottom-left with C label)
  - choice_D.png — Candidate D (bottom-right with D label)

Also generates split_exam.jsonl with split-image format fields:
  - images: [current.png, choice_A.png, choice_B.png, choice_C.png, choice_D.png]
  - prompt: Split-image prompt (numbered images 1/2/3/4/5)
  - label_map: {A: 1, B: 2, C: 3, D: 4} (correct label from original composite)

Original composite version (original_exam.jsonl) remains unchanged.

Usage:
  python split_taskA_frames.py [--exam_dir DATASETS_CANONICAL/taska] [--num 50]
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# Add config import
import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from config import MINIGRID_ROOT, TEMP_ROOT

# Canonical layout constants (from build_exam.py lines 454-471)
# Canvas: w*3 x h*2, left column is current (vertically centered), right 2x2 grid is candidates
CANVAS_W_COLS = 3   # 3 columns
CANVAS_H_ROWS = 2   # 2 rows

FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def split_composite(img: Image.Image, w: int, h: int) -> dict[str, Image.Image]:
    """
    Split a (w*3, h*2) composite image into 5 sub-images.

    Layout:
    +----------------+----------------+----------------+
    |                |       [A]      |       [B]      |
    |   CURRENT      |  candidate 0   |  candidate 1   |
    |  (centered)   |                |                |
    +----------------+----------------+----------------+
    |                |       [C]      |       [D]      |
    |                |  candidate 2   |  candidate 3  |
    +----------------+----------------+----------------+

    Current region: x=0, y=h//2, width=w, height=h (lower half of left column)
    """
    current = img.crop((0, h // 2, w, h // 2 + h))
    cand_a = img.crop((w, 0, w * 2, h))
    cand_b = img.crop((w * 2, 0, w * 3, h))
    cand_c = img.crop((w, h, w * 2, h * 2))
    cand_d = img.crop((w * 2, h, w * 3, h * 2))
    return {
        "current": current,
        "choice_A": cand_a,
        "choice_B": cand_b,
        "choice_C": cand_c,
        "choice_D": cand_d,
    }


def _label_image(img: Image.Image, letter: str, size: int = 20) -> Image.Image:
    """Add letter label to the top-left corner of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([4, 4, 30, 30], fill=(0, 0, 0))
    draw.text((8, 6), letter, fill=(255, 255, 255), font=load_font(size))
    return out


def _center_image_in_canvas(img: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    """Center an image within a canvas of specified size."""
    out = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    x = (canvas_w - img.width) // 2
    y = (canvas_h - img.height) // 2
    out.paste(img, (x, y))
    return out


def run(exam_dir: Path, out_dir: Path, num: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    exam_path = exam_dir / "task_a_exam.jsonl"
    if not exam_path.exists():
        print(f"Error: {exam_path} does not exist.")
        return

    rows = []
    with exam_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} exam items from {exam_path}")

    # Split images
    split_rows = []
    split_img_dir = out_dir / "images"
    split_img_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        rel_img_path = row["image"]
        src_img_path = exam_dir / rel_img_path
        if not src_img_path.exists():
            print(f"Warning: {src_img_path} not found, skipping.")
            continue

        img = Image.open(src_img_path).convert("RGB")
        w, h = img.size

        # Verify dimensions match expected composite layout
        if w % CANVAS_W_COLS != 0 or h % CANVAS_H_ROWS != 0:
            print(f"Warning: {rel_img_path} has unexpected dimensions {img.size}, skipping.")
            continue

        cell_w = w // CANVAS_W_COLS
        cell_h = h // CANVAS_H_ROWS

        parts = split_composite(img, cell_w, cell_h)

        task = row.get("uid", f"item_{i}").split(".")[1]
        group = row.get("uid", f"item_{i}").split(".")[2]
        split_task_dir = split_img_dir / task
        split_task_dir.mkdir(parents=True, exist_ok=True)

        # Save split images with letter labels
        rel_paths = {}
        for letter, part_img in parts.items():
            filename = f"{group}_{letter}.png"
            rel_path = f"{task}/{filename}"
            labeled = _label_image(part_img, letter[-1], size=20)
            labeled.save(split_task_dir / filename)
            rel_paths[letter] = rel_path

        # Update prompt for split format
        new_prompt = (
            "On the left is the current state (image 1). "
            "On the right are 4 candidates (images 2-5): A, B, C, D.\n"
            "Given the action below, select the correct next state.\n"
            f"Action: {row.get('action', 'unknown')}"
        )

        # Create label mapping: letter -> index in images array
        label_map = {"A": 1, "B": 2, "C": 3, "D": 4}

        new_row = dict(row)
        new_row["images"] = [rel_paths["current"]] + [
            rel_paths["choice_A"],
            rel_paths["choice_B"],
            rel_paths["choice_C"],
            rel_paths["choice_D"],
        ]
        new_row["prompt"] = new_prompt
        new_row["label_map"] = label_map
        new_row["label"] = label_map[row["answer"]]
        new_row["original_image"] = row["image"]
        new_row["original_label"] = row["label"]

        split_rows.append(new_row)

    # Write split exam JSONL
    split_exam_path = out_dir / "split_exam.jsonl"
    with split_exam_path.open("w", encoding="utf-8") as f:
        for row in split_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(split_rows)} items to {split_exam_path}")

    # Report format comparison
    orig_prompt_len = sum(len(r.get("prompt", "")) for r in rows[:len(split_rows)])
    new_prompt_len = sum(len(r.get("prompt", "")) for r in split_rows)
    print(f"\nFormat comparison (first {len(split_rows)} items):")
    print(f"  Original composite prompt avg length: {orig_prompt_len / len(split_rows):.1f} chars")
    print(f"  New split-image prompt avg length: {new_prompt_len / len(split_rows):.1f} chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Task A composite -> split frames converter")
    ap.add_argument(
        "--exam_dir",
        default=str(MINIGRID_ROOT / "taska"),
        help="Original canonical Task A exam directory",
    )
    ap.add_argument(
        "--out_dir",
        default=str(TEMP_ROOT / "taskA_image_format_comparison"),
        help="Output directory",
    )
    ap.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of questions to process (default 50)",
    )
    args = ap.parse_args()
    run(Path(args.exam_dir), Path(args.out_dir), args.num)

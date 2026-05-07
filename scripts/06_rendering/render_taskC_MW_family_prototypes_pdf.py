#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/mnt/c/Windows/Fonts/msyhbd.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]

PAGE_W = 2480
PAGE_H = 3508
MARGIN = 120

FAMILY_NAMES = {
    "doorkey": "DoorKey",
    "multiroom": "MultiRoom",
    "redblue": "RedBlue",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def fit_contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_cover(rows: list[dict[str, Any]], exam_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    body_font = load_font(42)
    draw.text((MARGIN, 360), "Task C-MW Prototype Review Pack", fill=(0, 0, 0), font=title_font)
    text = (
        f"Total items: {len(rows)}\n"
        f"Exam path: {exam_path}\n\n"
        "Design goal:\n"
        "This version is not continuous full-frame Task D, but rather a variable-K sparse keyframe Task C prototype retaining only key event frames and the last two frames.\n"
        "First validate whether sparse storyboards work for DoorKey / MultiRoom / RedBlue families.\n"
        "Prompts aligned with MiniGrid Task C task goal + Yes/No style.\n\n"
        "Current variants: full / nocue / cf\n"
        "Nocue note: re-rendered per frame with temporary evidence hiding, no post-processing occlusion used."
    )
    y = 560
    for line in text.split("\n"):
        draw.text((MARGIN, y), line, fill=(0, 0, 0), font=body_font)
        y += 58
    return page


def render_question_page(row: dict[str, Any], idx: int, total: int, root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(52)
    meta_font = load_font(32)
    steps_font = load_font(30)
    y = MARGIN
    draw.text((MARGIN, y), f"Prototype {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 74
    lines = [
        f"Family: {FAMILY_NAMES.get(str(row['family']), str(row['family']))}",
        f"Template: {row['template_id']}",
        f"Variant: {row['variant']}",
        f"K-frame: {row['k_frame']}",
        f"Prompt: {row.get('prompt_version', 'n/a')}",
    ]
    for line in lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font)
        y += 44
    task_goal = str(row.get("task_goal", ""))
    if task_goal:
        draw.text((MARGIN, y), task_goal, fill=(20, 20, 20), font=meta_font)
        y += 44
    if row.get("variant") == "nocue" and row.get("nocue_meta"):
        draw.text((MARGIN, y), f"Nocue: {row['nocue_meta'].get('strategy')}", fill=(20, 20, 20), font=meta_font)
        y += 44
    steps_text = f"selected_steps: {row['selected_steps']}"
    draw.rounded_rectangle((MARGIN, y, PAGE_W - MARGIN, y + 52), radius=14, fill=(245, 245, 245), outline=(210, 210, 210), width=2)
    draw.text((MARGIN + 16, y + 10), steps_text, fill=(0, 0, 0), font=steps_font)
    y += 68
    img = Image.open(root / row["image"]).convert("RGB")
    fitted = fit_contain(img, PAGE_W - 2 * MARGIN, PAGE_H - y - MARGIN)
    page.paste(fitted, ((PAGE_W - fitted.width) // 2, y + 20))
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description="Render review PDF for Task C-MW family prototypes")
    ap.add_argument(
        "--exam",
        default="tmp_miniworld/taskC-MW-family-prototypes/task_c_mw_family_prototypes.jsonl",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskC-MW-family-prototypes/taskC_MW_family_prototypes_review.pdf",
    )
    args = ap.parse_args()

    exam = Path(args.exam)
    output = Path(args.output)
    rows = load_rows(exam)

    pages = [render_cover(rows, exam)]
    for idx, row in enumerate(rows, start=1):
        pages.append(render_question_page(row, idx, len(rows), exam.parent))

    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output, "PDF", resolution=180.0, save_all=True, append_images=pages[1:])
    print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()

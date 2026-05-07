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
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def render_cover(rows: list[dict[str, Any]], summary_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(44)
    body_font = load_font(40)
    draw.text((MARGIN, 360), "Task D-MW High-Discriminability Expansion Review Pack", fill=(0, 0, 0), font=title_font)
    draw.text((MARGIN, 470), "High-Discriminability Expansion Round 1", fill=(40, 40, 40), font=sub_font)
    body = (
        f"Template count: {len(rows)}\n"
        f"Summary path: {summary_path}\n\n"
        "Design goal:\n"
        "This round does not pursue more decorations in scenes, but rather clearer visual separation between success/failure and correct/incorrect rule images.\n\n"
        "Reading guide:\n"
        "Each page shows full on top and cf on bottom, for direct comparison of whether high-discriminability structural templates truly bring stronger visual separation."
    )
    y = 680
    for line in body.split("\n"):
        draw.text((MARGIN, y), line, fill=(0, 0, 0), font=body_font)
        y += 58
    return page


def render_question_page(row: dict[str, Any], idx: int, total: int, root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(52)
    meta_font = load_font(32)
    section_font = load_font(40)

    y = MARGIN
    draw.text((MARGIN, y), f"Template {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 78
    draw.text((MARGIN, y), f"Family: {FAMILY_NAMES.get(str(row['family']), str(row['family']))}", fill=(20, 20, 20), font=meta_font)
    y += 42
    draw.text((MARGIN, y), f"Template ID: {row['template_id']}", fill=(20, 20, 20), font=meta_font)
    y += 42
    draw.text((MARGIN, y), f"Note: {row['note']}", fill=(20, 20, 20), font=meta_font)
    y += 52

    full_img = Image.open(root / row["full_storyboard_relpath"]).convert("RGB")
    cf_img = Image.open(root / row["cf_storyboard_relpath"]).convert("RGB")

    max_w = PAGE_W - 2 * MARGIN
    panel_h = (PAGE_H - y - MARGIN - 120) // 2

    draw.text((MARGIN, y), "Full / Success Trajectory", fill=(0, 0, 0), font=section_font)
    y += 54
    full_fit = fit_contain(full_img, max_w, panel_h - 54)
    page.paste(full_fit, ((PAGE_W - full_fit.width) // 2, y))
    y += panel_h

    draw.text((MARGIN, y), "CF / Failure Trajectory", fill=(0, 0, 0), font=section_font)
    y += 54
    cf_fit = fit_contain(cf_img, max_w, panel_h - 54)
    page.paste(cf_fit, ((PAGE_W - cf_fit.width) // 2, y))
    return page


def render_answer_page(rows: list[dict[str, Any]]) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    body_font = load_font(32)
    y = MARGIN
    draw.text((MARGIN, y), "Answer Appendix", fill=(0, 0, 0), font=title_font)
    y += 90
    for row in rows:
        lines = [
            f"{row['template_id']}",
            f"{FAMILY_NAMES.get(str(row['family']), str(row['family']))}",
            f"full = Success, cf = Fail, pair_ok = {row['pair_ok']}",
            "",
        ]
        for line in lines:
            draw.text((MARGIN, y), line, fill=(20, 20, 20), font=body_font)
            y += 40
        y += 18
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description="Render review PDF for high-discriminability MiniWorld round1 pack")
    ap.add_argument(
        "--summary",
        default="tmp_miniworld/taskD-MW-highdisc-expansion-round1-clean/highdisc_round1_summary.json",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskD-MW-highdisc-expansion-round1-clean/taskD_MW_highdisc_round1_review.pdf",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    output = Path(args.output)
    rows = load_rows(summary_path)
    root = summary_path.parent

    pages = [render_cover(rows, summary_path)]
    for idx, row in enumerate(rows, start=1):
        pages.append(render_question_page(row, idx, len(rows), root))
    pages.append(render_answer_page(rows))

    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output, "PDF", resolution=180.0, save_all=True, append_images=pages[1:])
    print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()

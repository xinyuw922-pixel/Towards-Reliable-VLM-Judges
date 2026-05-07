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
MARGIN = 110
SECTION_GAP = 36

FAMILY_NAMES = {
    "doorkey": "DoorKey",
    "multiroom": "MultiRoom",
    "redblue": "RedBlue",
}

VARIANT_NAMES = {
    "full": "Full / Complete Evidence",
    "nocue": "Nocue / No Cue",
    "cf": "CF / Counterfactual Failure",
}

TASK_GOAL_EN = {
    "doorkey": "Task goal: Pick up key, open door, and reach the goal.",
    "multiroom": "Task goal: Pass through two doors in order, and finally reach the goal.",
    "redblue": "Task goal: Open the red door first, then open the blue door.",
}

ANSWER_NAMES = {
    "Success": "Yes / Success",
    "Fail": "No / Failure",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def fit_contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def render_cover(rows: list[dict[str, Any]], exam_path: Path, mode: str) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(68)
    body_font = load_font(38)
    mode_label = "Question Only" if mode == "blind" else "With Answer Key"
    draw.text((MARGIN, 300), f"Task C-MW Formal Exam Pack Review ({mode_label})", fill=(0, 0, 0), font=title_font)
    text = (
        f"Exam path: {exam_path}\n"
        f"Total samples: {len(rows)}\n\n"
        "Contents:\n"
        "- Each sample on its own page for sequential human review\n"
        "- Main variants: full / nocue / cf\n"
        "- Keyframe selection: event-plus-tail-only-v1\n"
        "- Prompts aligned with MiniGrid Task C task goal + Yes/No style\n"
        "- Nocue uses environment-level temporary invisibility, no post-processing occlusion\n"
        + (
            "- Current mode: Question only - no ground truth or variant semantic labels shown"
            if mode == "blind"
            else "- Current mode: With answer key - shows ground truth, variant type, and sample metadata"
        )
    )
    y = 520
    for line in text.split("\n"):
        draw.text((MARGIN, y), line, fill=(0, 0, 0), font=body_font)
        y += 54
    return page


def render_question_page(row: dict[str, Any], idx: int, total: int, root: Path, mode: str) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(48)
    meta_font = load_font(32)
    body_font = load_font(34)

    y = MARGIN
    draw.text((MARGIN, y), f"Question {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 66
    lines = [f"Family: {FAMILY_NAMES.get(str(row['family']), str(row['family']))}"]
    if mode == "answer-key":
        lines.extend(
            [
                f"Template: {row['template_id']}",
                f"Difficulty: {row.get('difficulty', 'default')}",
                f"Variant: {VARIANT_NAMES.get(str(row['variant']), str(row['variant']))}",
                f"Answer: {ANSWER_NAMES.get(str(row['answer']), str(row['answer']))}",
                f"Selected steps: {row['selected_steps']}",
            ]
        )
        if row.get("nocue_meta"):
            lines.append(f"Nocue strategy: {row['nocue_meta'].get('strategy')}")
    lines.append(TASK_GOAL_EN.get(str(row["family"]), str(row.get("task_goal", ""))))
    lines.append("Please judge whether this trajectory completes the task goal. Answer: Yes or No.")
    for line in lines:
        if not line:
            continue
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font)
        y += 40

    img = Image.open(root / row["image"]).convert("RGB")
    fitted = fit_contain(img, PAGE_W - 2 * MARGIN, PAGE_H - y - MARGIN - 20)
    page.paste(fitted, ((PAGE_W - fitted.width) // 2, y + 20))

    return page


def main() -> None:
    ap = argparse.ArgumentParser(description="Render review PDF for Task C-MW formal exam")
    ap.add_argument(
        "--exam",
        default="tmp_miniworld/taskC-MW-formal-bank/task_c_mw_formal_exam.jsonl",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskC-MW-formal-bank/taskC_MW_formal_review.pdf",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="blind",
    )
    args = ap.parse_args()

    exam = Path(args.exam)
    rows = load_rows(exam)

    pages = [render_cover(rows, exam, args.mode)]
    for idx, row in enumerate(rows, start=1):
        pages.append(render_question_page(row, idx, len(rows), exam.parent, args.mode))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output, "PDF", resolution=180.0, save_all=True, append_images=pages[1:])
    print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()

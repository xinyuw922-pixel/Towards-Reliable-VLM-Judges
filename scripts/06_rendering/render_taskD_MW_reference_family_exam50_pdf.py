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

DIFFICULTY_NAMES = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
    "structural": "Structural",
    "structural_v2": "Structural v2",
    "highdisc_clear": "High-Disc",
    "highdisc_branch": "High-Disc",
    "highdisc_zigzag": "High-Disc",
}

SOURCE_NAMES = {
    "paired": "Base Migration",
    "structural": "Structural",
    "highdisc_round1": "High-Disc Round 1",
    "highdisc_round2": "High-Disc Round 2",
    "highdisc_round3": "High-Disc Round 3",
}


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        buf = ""
        for ch in para:
            test = buf + ch
            if draw.textlength(test, font=font) <= max_width:
                buf = test
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        if buf:
            lines.append(buf)
    return lines


def add_wrapped(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.ImageFont, max_width: int, line_gap: int) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_gap
    return y


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_family_name(family: str) -> str:
    return FAMILY_NAMES.get(family, family)


def get_difficulty_name(difficulty: str) -> str:
    return DIFFICULTY_NAMES.get(difficulty, difficulty)


def get_source_name(source: str) -> str:
    return SOURCE_NAMES.get(source, source)


def render_cover(records: list[dict[str, Any]], exam_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(40)

    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record["source_bucket"])
        source_counts[source] = source_counts.get(source, 0) + 1

    draw.text((MARGIN, 340), "Task D-MW-RF 50 Questions Review Pack", fill=(0, 0, 0), font=title_font)
    draw.text((MARGIN, 450), "Reference-Family MiniWorld Exam 50", fill=(40, 40, 40), font=sub_font)

    body = (
        f"Total items: {len(records)}\n"
        f"Exam path: {exam_path}\n\n"
        "Composition:\n"
        f"- Base migration: {source_counts.get('paired', 0)} items\n"
        f"- Structural: {source_counts.get('structural', 0)} items\n"
        f"- High-discriminability expansion: {source_counts.get('highdisc_round1', 0) + source_counts.get('highdisc_round2', 0) + source_counts.get('highdisc_round3', 0)} items\n\n"
        "Design rationale:\n"
        "For each family, provide the simplest known-success reference, then judge whether query completes the same task rules.\n"
        "This version focuses on high-discriminability templates, trying to avoid relying solely on clutter for difficulty."
    )
    add_wrapped(draw, MARGIN, 650, body, body_font, PAGE_W - 2 * MARGIN, 58)
    return page


def render_instruction_page() -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(58)
    body_font = load_font(38)
    draw.text((MARGIN, MARGIN), "Instructions", fill=(0, 0, 0), font=title_font)
    text = (
        "Each question consists of two parts:\n"
        "1. Top is Reference: a known-success example from the same family.\n"
        "2. Bottom is Query: trajectory with more complex path, structure, or counterfactual intervention.\n\n"
        "Your task:\n"
        "Judge whether the Query also completes the same task goal as the Reference.\n\n"
        "Answer requirement: Output only Success or Fail."
    )
    add_wrapped(draw, MARGIN, 240, text, body_font, PAGE_W - 2 * MARGIN, 54)
    return page


def render_question_page(record: dict[str, Any], idx: int, total: int, exam_root: Path, mode: str) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(52)
    meta_font = load_font(32)

    y = MARGIN
    draw.text((MARGIN, y), f"Question {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 80
    lines = [f"Family: {get_family_name(record['family'])}"]
    if mode == "answer-key":
        lines.extend(
            [
                f"Source: {get_source_name(str(record['source_bucket']))}",
                f"Reference: {record['reference_template_id']} / {get_difficulty_name(str(record['reference_difficulty']))}",
                f"Query: {record['query_template_id']} / {get_difficulty_name(str(record['query_difficulty']))}",
                f"Query variant: {record['query_variant']}",
                f"Ground truth: {record['answer']}",
            ]
        )
    lines.extend(
        [
            "",
            "Judge whether the bottom Query also completes the same task goal as the top Reference.",
            "Answer format: Success or Fail.",
        ]
    )
    for line in lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font)
        y += 46

    img = Image.open(exam_root / record["image"]).convert("RGB")
    max_w = PAGE_W - 2 * MARGIN
    max_h = PAGE_H - y - MARGIN
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    page.paste(img, ((PAGE_W - img.width) // 2, y + 20))
    return page


def render_answer_pages(records: list[dict[str, Any]]) -> list[Image.Image]:
    pages: list[Image.Image] = []
    title_font = load_font(54)
    body_font = load_font(30)
    chunk_size = 5
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        page, draw = create_blank_page()
        y = MARGIN
        draw.text((MARGIN, y), f"Answer Appendix {start // chunk_size + 1}", fill=(0, 0, 0), font=title_font)
        y += 90
        for record in chunk:
            block = (
                f"{record['exam_id']}\n"
                f"{get_family_name(record['family'])} | {get_source_name(str(record['source_bucket']))} | {record['query_template_id']} | {record['query_variant']}\n"
                f"Ground truth: {record['answer']}\n"
            )
            y = add_wrapped(draw, MARGIN, y, block, body_font, PAGE_W - 2 * MARGIN, 40)
            y += 20
            draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=(180, 180, 180), width=2)
            y += 20
        pages.append(page)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Render review PDF for 50-question MiniWorld reference-family exam")
    ap.add_argument(
        "--exam",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/task_d_mw_reference_family_exam50.jsonl",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/task_d_mw_reference_family_exam50_review.pdf",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="blind",
    )
    args = ap.parse_args()

    exam_path = Path(args.exam)
    output = Path(args.output)
    records = load_records(exam_path)
    exam_root = exam_path.parent

    pages = [render_cover(records, exam_path), render_instruction_page()]
    for idx, record in enumerate(records, start=1):
        pages.append(render_question_page(record, idx, len(records), exam_root, args.mode))
    if args.mode == "answer-key":
        pages.extend(render_answer_pages(records))

    output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output, "PDF", resolution=180.0, save_all=True, append_images=pages[1:])
    print(f"Saved PDF to {output}")


if __name__ == "__main__":
    main()

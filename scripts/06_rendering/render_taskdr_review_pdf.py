#!/usr/bin/env python3
"""
Render a PDF review pack for Task D-R manual auditing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/mnt/c/Windows/Fonts/msyh.ttc")
PAGE_W = 2480
PAGE_H = 3508
MARGIN = 120

TITLE = "Task D-R Review Pack"
SUBTITLE = "GridWM-Judge / Reference-Aided Task D Audit"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    return ImageFont.truetype(str(FONT_PATH), size=size)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines: List[str] = []
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


def add_wrapped_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    line_gap: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_gap
    return y


def load_records(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def env_name(env_task: str) -> str:
    mapping = {
        "doorkey": "DoorKey",
        "keycorridor": "KeyCorridor",
        "lavagap": "LavaGap",
        "memory": "Memory",
        "multiroom": "MultiRoom",
        "redblue": "RedBlueDoor",
    }
    return mapping.get(env_task, env_task)


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def prompt_text() -> str:
    return (
        "Task Instructions (Task D-R)\n\n"
        "Each question contains two trajectory fragments from the same environment.\n"
        "The top shows a reference trajectory with known success; the bottom shows the query trajectory to be judged.\n\n"
        "Your task: Determine whether the bottom query trajectory also completes the same task goal as the top reference.\n\n"
        "The extended version includes three types of variations:\n"
        "1. Query may come from full / nocue / cf variants\n"
        "2. Images may be clean or styled\n"
        "3. Prompts may have neu / pos / neg framing\n\n"
        "Answer requirements:\n"
        "1. Reference trajectories are always successful examples.\n"
        "2. Judge only based on images whether query completes the same task.\n"
        "3. Do not treat style as semantic change; do not be biased by framing cues.\n"
        "4. Output only: Success or Fail."
    )


def render_cover(records: List[Dict[str, Any]]) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(46)

    y = 420
    draw.text((MARGIN, y), TITLE, fill=(0, 0, 0), font=title_font)
    y += 110
    draw.text((MARGIN, y), SUBTITLE, fill=(40, 40, 40), font=sub_font)
    y += 120

    body = (
        f"Total items: {len(records)}\n"
        "Purpose: Use a known-successful reference trajectory to help humans and models judge whether the query trajectory completes the same task.\n"
        f"Contents: Cover + Instructions + {len(records)} question pages + Answer appendix.\n\n"
        "Suggested approach:\n"
        "1. First examine the top reference trajectory to understand what counts as success in this environment.\n"
        "2. Then examine the bottom query trajectory to judge whether it achieves the same goal.\n"
        "3. Also pay attention to query variant / visual / framing labels.\n"
        "4. Finally check against the answer appendix."
    )
    add_wrapped_block(draw, MARGIN, y, body, body_font, PAGE_W - 2 * MARGIN, 64)
    return page


def render_instruction_page() -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(58)
    body_font = load_font(36)

    y = MARGIN
    draw.text((MARGIN, y), "Instructions", fill=(0, 0, 0), font=title_font)
    y += 90
    add_wrapped_block(draw, MARGIN, y, prompt_text(), body_font, PAGE_W - 2 * MARGIN, 50)
    return page


def render_question_page(record: Dict[str, Any], idx: int, total: int, exam_root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    meta_font = load_font(34)
    body_font = load_font(38)

    env = record["env_id"]
    y = MARGIN
    draw.text((MARGIN, y), f"Question {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 78

    meta_lines = [
        f"Environment: {env_name(env)}",
        f"Reference group: {record['reference_group_id']}",
        f"Query group: {record['group_id']}",
        f"Query variant: {record['query_variant']}",
        f"Visual: {record.get('visual', 'clean')}",
        f"Framing: {record.get('framing', 'neu')}",
        f"Exam ID: {record['exam_id']}",
        "Please judge whether the bottom query trajectory completes the same task goal as the top reference trajectory.",
        "Answer format: Success or Fail.",
    ]
    for line in meta_lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font if "Please judge" not in line and "Answer format" not in line else body_font)
        y += 48 if "Please judge" not in line and "Answer format" not in line else 56

    img_path = exam_root / record["image"]
    img = Image.open(img_path).convert("RGB")
    max_w = PAGE_W - 2 * MARGIN
    max_h = PAGE_H - y - MARGIN
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    x = (PAGE_W - img.width) // 2
    page.paste(img, (x, y + 20))
    return page


def answer_summary(record: Dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Exam ID: {record['exam_id']}",
            f"Environment: {env_name(record['env_id'])}",
            f"Reference: {record['reference_group_id']}",
            f"Query: {record['group_id']} / {record['query_variant']}",
            f"Visual: {record.get('visual', 'clean')}",
            f"Framing: {record.get('framing', 'neu')}",
            f"Ground truth: {record['answer']}",
            "",
        ]
    )


def render_answer_pages(records: List[Dict[str, Any]]) -> List[Image.Image]:
    pages: List[Image.Image] = []
    title_font = load_font(54)
    body_font = load_font(30)
    chunk_size = 4

    for page_idx in range(0, len(records), chunk_size):
        chunk = records[page_idx:page_idx + chunk_size]
        page, draw = create_blank_page()
        y = MARGIN
        draw.text(
            (MARGIN, y),
            f"Answer Appendix {page_idx // chunk_size + 1}/{(len(records) + chunk_size - 1) // chunk_size}",
            fill=(0, 0, 0),
            font=title_font,
        )
        y += 80
        for record in chunk:
            y = add_wrapped_block(draw, MARGIN, y, answer_summary(record), body_font, PAGE_W - 2 * MARGIN, 40)
            y += 20
            draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=(180, 180, 180), width=2)
            y += 20
        pages.append(page)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exam",
        type=Path,
        default=Path("outputs/exams_taskd/task_d_exam.jsonl"),
        help="Task D-R exam JSONL path",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/exams_taskd/task_d_r_review.pdf"),
        help="Output PDF path",
    )
    args = ap.parse_args()

    records = load_records(args.exam)
    if not records:
        raise ValueError(f"No records found in {args.exam}")

    exam_root = args.exam.parent
    pages: List[Image.Image] = [
        render_cover(records),
        render_instruction_page(),
    ]
    for idx, record in enumerate(records, start=1):
        pages.append(render_question_page(record, idx, len(records), exam_root))
    pages.extend(render_answer_pages(records))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        args.output,
        "PDF",
        resolution=200.0,
        save_all=True,
        append_images=pages[1:],
    )
    print(args.output)
    print(f"pages={len(pages)}")


if __name__ == "__main__":
    main()

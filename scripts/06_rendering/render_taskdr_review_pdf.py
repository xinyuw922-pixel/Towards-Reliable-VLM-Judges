#!/usr/bin/env python3
"""
Render a Chinese PDF review pack for Task D-R manual auditing.
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

TITLE = "Task D-R 人工审阅包"
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


def env_name_zh(env_task: str) -> str:
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


def prompt_translation() -> str:
    return (
        "任务说明（Task D-R 中文翻译）\n\n"
        "每道题包含两段来自同一环境的轨迹拼图。\n"
        "上方是已知成功的参考轨迹，下方是待判断轨迹。\n\n"
        "你的任务是判断：下方 query 轨迹是否也完成了与上方 reference 相同的任务目标。\n\n"
        "当前扩展版还包含三类变化：\n"
        "1. query 可能来自 full / nocue / cf 三种 variant\n"
        "2. 图像可能是 clean 或 style\n"
        "3. 题面可能带 neu / pos / neg 三种提示语气\n\n"
        "答题要求：\n"
        "1. 参考轨迹一定是成功示例。\n"
        "2. 只根据图像判断 query 是否完成相同任务。\n"
        "3. 不要把 style 当成语义变化，也不要被语气提示带偏。\n"
        "4. 只能输出 Success 或 Fail。"
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
        f"题目总数：{len(records)}\n"
        "用途：用一个已知成功参考轨迹，帮助人工和模型判断 query 轨迹是否也完成同一任务。\n"
        f"组成：封面 + 说明页 + {len(records)} 道题目页 + 答案附录。\n\n"
        "建议做法：\n"
        "1. 先看上半部分参考轨迹，理解这个环境里什么算成功。\n"
        "2. 再看下半部分 query 轨迹，判断它是否达到相同目标。\n"
        "3. 同时留意题目里的 query variant / visual / framing 标注。\n"
        "4. 最后再翻答案附录核对。"
    )
    add_wrapped_block(draw, MARGIN, y, body, body_font, PAGE_W - 2 * MARGIN, 64)
    return page


def render_instruction_page() -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(58)
    body_font = load_font(36)

    y = MARGIN
    draw.text((MARGIN, y), "说明页", fill=(0, 0, 0), font=title_font)
    y += 90
    add_wrapped_block(draw, MARGIN, y, prompt_translation(), body_font, PAGE_W - 2 * MARGIN, 50)
    return page


def render_question_page(record: Dict[str, Any], idx: int, total: int, exam_root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    meta_font = load_font(34)
    body_font = load_font(38)

    env = record["env_id"]
    y = MARGIN
    draw.text((MARGIN, y), f"题目 {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 78

    meta_lines = [
        f"环境：{env_name_zh(env)}",
        f"reference group：{record['reference_group_id']}",
        f"query group：{record['group_id']}",
        f"query variant：{record['query_variant']}",
        f"visual：{record.get('visual', 'clean')}（{record.get('visual_translation_zh', '')}）",
        f"framing：{record.get('framing', 'neu')}（{record.get('framing_translation_zh', '')}）",
        f"exam_id：{record['exam_id']}",
        "请判断下方 query 轨迹是否完成了与上方参考轨迹相同的任务目标。",
        "答案只能写：Success 或 Fail。",
    ]
    for line in meta_lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font if "请判断" not in line and "答案只能写" not in line else body_font)
        y += 48 if "请判断" not in line and "答案只能写" not in line else 56

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
            f"exam_id：{record['exam_id']}",
            f"环境：{env_name_zh(record['env_id'])}",
            f"reference：{record['reference_group_id']}",
            f"query：{record['group_id']} / {record['query_variant']}",
            f"visual：{record.get('visual', 'clean')}",
            f"framing：{record.get('framing', 'neu')}",
            f"标准答案：{record['answer']}",
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
            f"答案附录 {page_idx // chunk_size + 1}/{(len(records) + chunk_size - 1) // chunk_size}",
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
        default=Path("outputs/exams_taskd/task_d_r_review_zh.pdf"),
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

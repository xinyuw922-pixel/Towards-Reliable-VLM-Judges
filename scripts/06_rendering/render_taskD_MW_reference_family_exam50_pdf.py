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

FAMILY_ZH = {
    "doorkey": "DoorKey / 钥匙开门",
    "multiroom": "MultiRoom / 多房间",
    "redblue": "RedBlue / 红蓝门",
}

DIFFICULTY_ZH = {
    "easy": "简单",
    "medium": "中等",
    "hard": "较难",
    "structural": "结构增强",
    "structural_v2": "结构增强 v2",
    "highdisc_clear": "高区分度",
    "highdisc_branch": "高区分度",
    "highdisc_zigzag": "高区分度",
}

SOURCE_ZH = {
    "paired": "基础迁移",
    "structural": "结构增强",
    "highdisc_round1": "高区分度 round1",
    "highdisc_round2": "高区分度 round2",
    "highdisc_round3": "高区分度 round3",
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


def zh_family(family: str) -> str:
    return FAMILY_ZH.get(family, family)


def zh_difficulty(difficulty: str) -> str:
    return DIFFICULTY_ZH.get(difficulty, difficulty)


def zh_source(source: str) -> str:
    return SOURCE_ZH.get(source, source)


def render_cover(records: list[dict[str, Any]], exam_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(40)

    source_counts: dict[str, int] = {}
    for record in records:
        source = str(record["source_bucket"])
        source_counts[source] = source_counts.get(source, 0) + 1

    draw.text((MARGIN, 340), "Task D-MW-RF 50 题中文审阅包", fill=(0, 0, 0), font=title_font)
    draw.text((MARGIN, 450), "Reference-Family MiniWorld Exam 50", fill=(40, 40, 40), font=sub_font)

    body = (
        f"题目总数：{len(records)}\n"
        f"题库路径：{exam_path}\n\n"
        "构成：\n"
        f"- 基础迁移：{source_counts.get('paired', 0)} 题\n"
        f"- 结构增强：{source_counts.get('structural', 0)} 题\n"
        f"- 高区分度扩题：{source_counts.get('highdisc_round1', 0) + source_counts.get('highdisc_round2', 0) + source_counts.get('highdisc_round3', 0)} 题\n\n"
        "设计思路：\n"
        "每个 family 先给一个最简单的已知成功 reference，再判断 query 是否完成相同任务规则。\n"
        "这版重点加入了高区分度模板，尽量避免只靠 clutter 拉难度。"
    )
    add_wrapped(draw, MARGIN, 650, body, body_font, PAGE_W - 2 * MARGIN, 58)
    return page


def render_instruction_page() -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(58)
    body_font = load_font(38)
    draw.text((MARGIN, MARGIN), "说明页", fill=(0, 0, 0), font=title_font)
    text = (
        "每道题由上下两部分组成：\n"
        "1. 上方是 Reference：同一 family 的一个已知成功例子。\n"
        "2. 下方是 Query：路径更复杂、结构更复杂或规则被反事实干预的轨迹。\n\n"
        "你的任务是判断：\n"
        "Query 是否也完成了与 Reference 相同的任务目标。\n\n"
        "答题要求：只能输出 Success 或 Fail。"
    )
    add_wrapped(draw, MARGIN, 240, text, body_font, PAGE_W - 2 * MARGIN, 54)
    return page


def render_question_page(record: dict[str, Any], idx: int, total: int, exam_root: Path, mode: str) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(52)
    meta_font = load_font(32)

    y = MARGIN
    draw.text((MARGIN, y), f"题目 {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 80
    lines = [f"family：{zh_family(record['family'])}"]
    if mode == "answer-key":
        lines.extend(
            [
                f"来源：{zh_source(str(record['source_bucket']))}",
                f"reference：{record['reference_template_id']} / {zh_difficulty(str(record['reference_difficulty']))}",
                f"query：{record['query_template_id']} / {zh_difficulty(str(record['query_difficulty']))}",
                f"query variant：{record['query_variant']}",
                f"标准答案：{record['answer']}",
            ]
        )
    lines.extend(
        [
            "",
            "判断下方 Query 是否也完成了与上方 Reference 相同的任务目标。",
            "答案只能写：Success 或 Fail。",
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
        draw.text((MARGIN, y), f"答案附录 {start // chunk_size + 1}", fill=(0, 0, 0), font=title_font)
        y += 90
        for record in chunk:
            block = (
                f"{record['exam_id']}\n"
                f"{zh_family(record['family'])} | {zh_source(str(record['source_bucket']))} | {record['query_template_id']} | {record['query_variant']}\n"
                f"标准答案：{record['answer']}\n"
            )
            y = add_wrapped(draw, MARGIN, y, block, body_font, PAGE_W - 2 * MARGIN, 40)
            y += 20
            draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=(180, 180, 180), width=2)
            y += 20
        pages.append(page)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description="Render Chinese review PDF for 50-question MiniWorld reference-family exam")
    ap.add_argument(
        "--exam",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/task_d_mw_reference_family_exam50.jsonl",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskD-MW-reference-family-exam50/task_d_mw_reference_family_exam50_review_zh.pdf",
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

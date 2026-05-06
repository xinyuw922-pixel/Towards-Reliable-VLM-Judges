#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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

FAMILY_ZH = {
    "doorkey": "DoorKey / 钥匙开门",
    "multiroom": "MultiRoom / 多房间",
    "redblue": "RedBlue / 红蓝门",
}

VARIANT_ZH = {
    "full": "full / 完整证据",
    "nocue": "nocue / 去线索",
    "cf": "cf / 反事实失败",
}

TASK_GOAL_ZH = {
    "doorkey": "任务目标：拿到钥匙，打开门，并到达目标格。",
    "multiroom": "任务目标：依次通过两扇门，并最终到达目标格。",
    "redblue": "任务目标：先打开红门，再打开蓝门。",
}

ANSWER_ZH = {
    "Success": "是 / 成功",
    "Fail": "否 / 失败",
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
    mode_zh = "题目版" if mode == "blind" else "答案版"
    draw.text((MARGIN, 300), f"Task C-MW 正式题库审阅包（{mode_zh}）", fill=(0, 0, 0), font=title_font)
    text = (
        f"题库路径：{exam_path}\n"
        f"样本总数：{len(rows)}\n\n"
        "内容：\n"
        "- 每条样本单独成页，适合人工顺序审题\n"
        "- 主线变体为 full / nocue / cf\n"
        "- 关键帧选择使用 event-plus-tail-only-v1\n"
        "- 提示词对齐 MiniGrid Task C 的 Task goal + Yes/No 风格\n"
        "- nocue 采用环境级临时不可见，不使用后处理遮挡\n"
        + (
            "- 当前为题目版：不显示标准答案，也不显示变体语义标签"
            if mode == "blind"
            else "- 当前为答案版：显示标准答案、变体类型与样本元数据"
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
    draw.text((MARGIN, y), f"题目 {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 66
    lines = [f"family：{FAMILY_ZH.get(str(row['family']), str(row['family']))}"]
    if mode == "answer-key":
        lines.extend(
            [
                f"template：{row['template_id']}",
                f"difficulty：{row.get('difficulty', 'default')}",
                f"variant：{VARIANT_ZH.get(str(row['variant']), str(row['variant']))}",
                f"answer：{ANSWER_ZH.get(str(row['answer']), str(row['answer']))}",
                f"selected_steps：{row['selected_steps']}",
            ]
        )
        if row.get("nocue_meta"):
            lines.append(f"nocue strategy：{row['nocue_meta'].get('strategy')}")
    lines.append(TASK_GOAL_ZH.get(str(row["family"]), str(row.get("task_goal", ""))))
    lines.append("请判断该轨迹是否完成了任务目标。作答只能写：是 或 否。")
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
    ap = argparse.ArgumentParser(description="Render Chinese review PDF for Task C-MW formal exam")
    ap.add_argument(
        "--exam",
        default="tmp_miniworld/taskC-MW-formal-bank/task_c_mw_formal_exam.jsonl",
    )
    ap.add_argument(
        "--output",
        default="tmp_miniworld/taskC-MW-formal-bank/taskC_MW_formal_review_zh.pdf",
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

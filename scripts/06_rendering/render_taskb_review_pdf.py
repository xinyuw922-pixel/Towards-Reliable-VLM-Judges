#!/usr/bin/env python3
"""
Render a Chinese PDF review pack for Task B manual auditing.

Output structure:
1. Cover page
2. One instruction page with translated Task B v7 prompt
3. One page per question (image + short Chinese task statement)
4. Answer appendix pages at the end
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/mnt/c/Windows/Fonts/msyh.ttc")
PAGE_W = 2480
PAGE_H = 3508
MARGIN = 120

TITLE = "Task B 人工做题审阅包"
SUBTITLE = "GridWM-Judge / Structured State Perception"


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


def color_zh(color: str | None) -> str | None:
    mapping = {
        "red": "红",
        "green": "绿",
        "blue": "蓝",
        "yellow": "黄",
        "purple": "紫",
        "grey": "灰",
    }
    if color is None:
        return None
    return mapping.get(color, color)


def type_zh(kind: str | None) -> str | None:
    mapping = {
        "empty": "空格",
        "wall": "墙",
        "door": "门",
        "goal": "目标格",
        "ball": "球",
        "key": "钥匙",
        "lava": "岩浆",
        "box": "箱子",
    }
    if kind is None:
        return None
    return mapping.get(kind, kind)


def door_state_zh(state: Any) -> str:
    mapping = {
        0: "0（开）",
        1: "1（关，未锁）",
        2: "2（锁住）",
        None: "null",
    }
    return mapping.get(state, str(state))


def carrying_zh(carrying: Dict[str, Any] | None) -> str:
    if not carrying:
        return "null（未携带）"
    return f"{type_zh(carrying.get('type'))}-{color_zh(carrying.get('color'))}"


def prompt_translation() -> str:
    return (
        "任务说明（Task B v7.1 中文翻译）\n\n"
        "你需要只根据这张 7x7 MiniGrid 图像，写出一个 JSON 对象，且只能包含 agent、front_cell、objects 三个键。\n\n"
        "坐标系：\n"
        "1. 使用基于图像的 1-based 坐标。\n"
        "2. 原点在图像左下角。\n"
        "3. x 向右增大，y 向上增大。\n"
        "4. 所有位置都写成 [x=列, y=行]。\n"
        "5. 合法范围是 x∈[1,7], y∈[1,7]。\n\n"
        "颜色词表：\n"
        "只允许使用 red, green, blue, yellow, purple, grey 这六种标准颜色；不要自造颜色名。\n"
        "岩浆必须是红色（red）。目标必须是绿色（green）。\n\n"
        "agent.dir 编码：\n"
        "0=右/east，1=下/south，2=左/west，3=上/north。\n\n"
        "state 字段——门状态视觉图例（关键）：\n"
        "front_cell 和每个 object 都必须有 state。\n"
        "只有门（door）需要用数字表示 state，从门的可见外观来判断：\n"
        "  state = 0（开）：门是敞开的，门板不封住门框，看起来可以通过\n"
        "  state = 1（关）：门把手标记看起来是一个小圆圈\n"
        "  state = 2（锁）：门把手标记看起来是一条短横线（\"-\"）\n"
        "根据门的实际外观来判断 state，不要根据任务上下文猜测。\n"
        "所有非门类型，state 一律写 null。\n"
        "不要对非门类型输出 0。\n"
        "不要输出 \"open\"、\"closed\" 等字符串。\n\n"
        "其他规则：\n"
        "1. 只能根据图片判断，不要依赖额外信息。\n"
        "2. 不要假设固定的 agent 位置。\n"
        "3. front_cell 必须单独报告。\n"
        "4. front_cell.type 是智能体正前方那个格子的占据类型。\n"
        "5. front_cell.type 只能用：empty, wall, door, goal, ball, key, lava, box。\n"
        "6. 不要把 front_cell 格子重复放进 objects。\n"
        "7. objects 里只报告非背景物体：key, ball, door, goal, lava, box。\n"
        "8. 不要报告 wall、floor、empty 作为 objects。\n"
        "9. 看不清楚的物体不要乱报；不要臆造物体、颜色或状态。\n\n"
        "建议答题格式：\n"
        "{\n"
        '  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},\n'
        '  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},\n'
        '  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]\n'
        "}\n\n"
        "只输出原始 JSON，不要 markdown、不要代码块、不要解释。\n\n"
        "提示词版本：Task B prompt v7.1（新增门状态视觉图例）\n"
        "本 PDF 前半部分不放答案，便于你先人工作答；最后附标准答案方便核对。"
    )


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


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
        "用途：人工阅读图片并手动判断 Task B 的结构化状态答案。\n"
        f"组成：封面 + 说明页 + {len(records)} 道题目页 + 答案附录。\n\n"
        "建议做法：\n"
        "1. 先只看题目页自己作答。\n"
        "2. 做完后再翻到最后的答案附录核对。\n"
        "3. 如果你觉得某题图像其实很清楚，就能顺便检验 Task B 的人类可读性。"
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
    add_wrapped_block(draw, MARGIN, y, prompt_translation(), body_font, PAGE_W - 2 * MARGIN, 48)
    return page


def render_question_page(record: Dict[str, Any], idx: int, total: int, exam_root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    meta_font = load_font(34)
    body_font = load_font(38)

    y = MARGIN
    draw.text((MARGIN, y), f"题目 {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 78
    meta_lines = [
        f"环境：{env_name_zh(record['uid'].split('.')[1])}",
        f"exam_id：{record['exam_id']}",
        "请根据下图，手动写出 Task B 的 JSON 答案。",
        "提示：先判断 agent，再判断 front_cell，最后列出 objects。",
    ]
    for line in meta_lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font if "exam_id" in line or "环境" in line else body_font)
        y += 48 if "exam_id" in line or "环境" in line else 56

    y += 20
    prompt = (
        "中文题面：请只根据图片，输出一个 JSON 对象，包含 agent、front_cell、objects。"
        " 坐标使用左下角为原点的 1-based [x, y]；agent.dir 使用 0=右, 1=下, 2=左, 3=上；"
        " door 的 state 用 0=开, 1=关, 2=锁，非 door 的 state 写 null。"
    )
    y = add_wrapped_block(draw, MARGIN, y, prompt, body_font, PAGE_W - 2 * MARGIN, 52)

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
    ans = record["answer_json"]
    agent = ans["agent"]
    fc = ans["front_cell"]
    objects = ans["objects"]

    lines = [
        f"exam_id：{record['exam_id']}",
        f"环境：{env_name_zh(record['uid'].split('.')[1])}",
        f"agent.pos = {agent['pos']}",
        f"agent.dir = {agent['dir']}（0右 / 1下 / 2左 / 3上）",
        f"agent.carrying = {carrying_zh(agent.get('carrying'))}",
        f"front_cell.pos = {fc['pos']}",
        f"front_cell.type = {fc['type']}（{type_zh(fc['type'])}）",
        f"front_cell.state = {door_state_zh(fc.get('state'))}",
        f"objects 数量 = {len(objects)}",
    ]
    if objects:
        for i, obj in enumerate(objects, start=1):
            lines.append(
                f"  {i}. {obj['type']}（{type_zh(obj['type'])}）, "
                f"pos={obj['pos']}, color={obj.get('color')}（{color_zh(obj.get('color'))}）, "
                f"state={door_state_zh(obj.get('state'))}"
            )
    else:
        lines.append("  无 objects")
    lines.append("")
    lines.append("原始标准答案 JSON：")
    lines.append(json.dumps(ans, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def render_answer_pages(records: List[Dict[str, Any]]) -> List[Image.Image]:
    pages: List[Image.Image] = []
    title_font = load_font(54)
    body_font = load_font(30)

    chunks: List[List[Dict[str, Any]]] = []
    chunk_size = 2
    for i in range(0, len(records), chunk_size):
        chunks.append(records[i:i + chunk_size])

    for page_idx, chunk in enumerate(chunks, start=1):
        page, draw = create_blank_page()
        y = MARGIN
        draw.text((MARGIN, y), f"答案附录 {page_idx}/{len(chunks)}", fill=(0, 0, 0), font=title_font)
        y += 80
        for record in chunk:
            y = add_wrapped_block(
                draw,
                MARGIN,
                y,
                answer_summary(record),
                body_font,
                PAGE_W - 2 * MARGIN,
                40,
            )
            y += 36
            draw.line((MARGIN, y, PAGE_W - MARGIN, y), fill=(180, 180, 180), width=2)
            y += 36
        pages.append(page)
    return pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exam",
        type=Path,
        default=Path("datasets/exams_taskb_repaired_candidate/task_b_exam.jsonl"),
        help="Task B exam JSONL path",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/exams_taskb_repaired_candidate/task_b_review_zh_with_answers.pdf"),
        help="Output PDF path",
    )
    args = ap.parse_args()

    records = load_records(args.exam)
    if not records:
        raise ValueError(f"No records found in {args.exam}")

    exam_root = args.exam.parent
    pages: List[Image.Image] = []
    pages.append(render_cover(records))
    pages.append(render_instruction_page())
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

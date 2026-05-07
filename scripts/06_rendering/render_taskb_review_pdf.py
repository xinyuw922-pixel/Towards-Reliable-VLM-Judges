#!/usr/bin/env python3
"""
Render a PDF review pack for Task B manual auditing.

Output structure:
1. Cover page
2. One instruction page with Task B v7 prompt
3. One page per question (image + short task statement)
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

TITLE = "Task B Manual Review Pack"
SUBTITLE = "GridWM-Judge / Structured State Perception"


def load_font(size: int) -> ImageFont.FreeTypeFont:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Font not found: {FONT_PATH}")
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


def color_name(color: str | None) -> str | None:
    mapping = {
        "red": "Red",
        "green": "Green",
        "blue": "Blue",
        "yellow": "Yellow",
        "purple": "Purple",
        "grey": "Grey",
    }
    if color is None:
        return None
    return mapping.get(color, color)


def type_name(kind: str | None) -> str | None:
    mapping = {
        "empty": "Empty",
        "wall": "Wall",
        "door": "Door",
        "goal": "Goal",
        "ball": "Ball",
        "key": "Key",
        "lava": "Lava",
        "box": "Box",
    }
    if kind is None:
        return None
    return mapping.get(kind, kind)


def door_state_name(state: Any) -> str:
    mapping = {
        0: "0 (Open)",
        1: "1 (Closed)",
        2: "2 (Locked)",
        None: "null",
    }
    return mapping.get(state, str(state))


def carrying_name(carrying: Dict[str, Any] | None) -> str:
    if not carrying:
        return "null (Not carrying)"
    return f"{type_name(carrying.get('type'))}-{color_name(carrying.get('color'))}"


def prompt_text() -> str:
    return (
        "Task Instructions (Task B v7.1)\n\n"
        "Based on this 7x7 MiniGrid image, write a JSON object with only three keys: agent, front_cell, objects.\n\n"
        "Coordinate system:\n"
        "1. Use image-based 1-based coordinates.\n"
        "2. Origin is bottom-left corner of the image.\n"
        "3. x increases to the right, y increases upward.\n"
        "4. All positions written as [x=col, y=row].\n"
        "5. Valid range is x in [1,7], y in [1,7].\n\n"
        "Color vocabulary:\n"
        "Only allow red, green, blue, yellow, purple, grey six standard colors; do not invent color names.\n"
        "Lava must be red. Goal must be green.\n\n"
        "agent.dir encoding:\n"
        "0=right/east, 1=down/south, 2=left/west, 3=up/north.\n\n"
        "state field -- Door state visual legend (critical):\n"
        "Both front_cell and each object must have state.\n"
        "Only doors need numeric state, judged from visible appearance:\n"
        "  state = 0 (Open): Door is open, frame not blocking passage\n"
        "  state = 1 (Closed): Door handle appears as a small circle\n"
        "  state = 2 (Locked): Door handle appears as a short dash ('-')\n"
        "Judge state from actual door appearance, not from task context.\n"
        "All non-door types, state must be null.\n"
        "Do not output 0 for non-door types.\n"
        "Do not output strings like 'open', 'closed'.\n\n"
        "Other rules:\n"
        "1. Judge only from image, do not rely on extra information.\n"
        "2. Do not assume fixed agent position.\n"
        "3. front_cell must be reported separately.\n"
        "4. front_cell.type is the cell type directly in front of the agent.\n"
        "5. front_cell.type can only be: empty, wall, door, goal, ball, key, lava, box.\n"
        "6. Do not put front_cell in objects.\n"
        "7. Only report non-background objects: key, ball, door, goal, lava, box.\n"
        "8. Do not report wall, floor, empty as objects.\n"
        "9. Do not guess objects you cannot see clearly; do not invent objects, colors, or states.\n\n"
        "Suggested answer format:\n"
        "{\n"
        '  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},\n'
        '  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},\n'
        '  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]\n'
        "}\n\n"
        "Output only raw JSON, no markdown, no code blocks, no explanations.\n"
        "Prompt version: Task B prompt v7.1 (new door state visual legend)\n"
        "First half of this PDF has no answers, for you to practice; answer key at the end for verification."
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
        f"Total items: {len(records)}\n"
        "Purpose: Human reading images and manually determining Task B structured state answers.\n"
        f"Contents: Cover + Instructions + {len(records)} question pages + Answer appendix.\n\n"
        "Suggested approach:\n"
        "1. First answer questions on your own.\n"
        "2. Check against answer appendix at the end.\n"
        "3. If you think some images are clear, this verifies Task B human readability."
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
    add_wrapped_block(draw, MARGIN, y, prompt_text(), body_font, PAGE_W - 2 * MARGIN, 48)
    return page


def render_question_page(record: Dict[str, Any], idx: int, total: int, exam_root: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    meta_font = load_font(34)
    body_font = load_font(38)

    y = MARGIN
    draw.text((MARGIN, y), f"Question {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 78
    meta_lines = [
        f"Environment: {env_name(record['uid'].split('.')[1])}",
        f"Exam ID: {record['exam_id']}",
        "Based on the image below, manually write Task B JSON answer.",
        "Hint: Judge agent first, then front_cell, finally list objects.",
    ]
    for line in meta_lines:
        draw.text((MARGIN, y), line, fill=(20, 20, 20), font=meta_font if "Exam ID" in line or "Environment" in line else body_font)
        y += 48 if "Exam ID" in line or "Environment" in line else 56

    y += 20
    prompt = (
        "English prompt: Based only on the image, output a JSON object with agent, front_cell, objects."
        " Coordinates use bottom-left origin 1-based [x, y]; agent.dir uses 0=right, 1=down, 2=left, 3=up;"
        " door state uses 0=open, 1=closed, 2=locked, non-door state is null."
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
        f"Exam ID: {record['exam_id']}",
        f"Environment: {env_name(record['uid'].split('.')[1])}",
        f"agent.pos = {agent['pos']}",
        f"agent.dir = {agent['dir']} (0=right / 1=down / 2=left / 3=up)",
        f"agent.carrying = {carrying_name(agent.get('carrying'))}",
        f"front_cell.pos = {fc['pos']}",
        f"front_cell.type = {fc['type']} ({type_name(fc['type'])})",
        f"front_cell.state = {door_state_name(fc.get('state'))}",
        f"objects count = {len(objects)}",
    ]
    if objects:
        for i, obj in enumerate(objects, start=1):
            lines.append(
                f"  {i}. {obj['type']} ({type_name(obj['type'])}), "
                f"pos={obj['pos']}, color={obj.get('color')} ({color_name(obj.get('color'))}), "
                f"state={door_state_name(obj.get('state'))}"
            )
    else:
        lines.append("  No objects")
    lines.append("")
    lines.append("Original ground truth JSON:")
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
        draw.text((MARGIN, y), f"Answer Appendix {page_idx}/{len(chunks)}", fill=(0, 0, 0), font=title_font)
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
        default=Path("datasets/exams_taskb_repaired_candidate/task_b_review_with_answers.pdf"),
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

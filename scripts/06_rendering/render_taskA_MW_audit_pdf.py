#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = [
    Path("/mnt/c/Windows/Fonts/msyh.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
PAGE_W = 2000
PAGE_H = 2828
MARGIN = 90


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


def add_wrapped_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_gap: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_gap
    return y


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / img.width, max_h / img.height, 1.0)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def split_concat_vertical(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    mid = img.height // 2
    top = img.crop((0, 0, img.width, mid))
    bottom = img.crop((0, mid, img.width, img.height))
    return top, bottom


def en_dir(dir_name: str) -> str:
    return {
        "N": "North",
        "E": "East",
        "S": "South",
        "W": "West",
    }.get(dir_name, dir_name)


def en_answer(label: str) -> str:
    return {
        "A": "A (can move forward to new position)",
        "B": "B (blocked by wall, stays in place)",
    }.get(label, label)


def build_cover(records: list[dict[str, Any]], exam_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(76)
    sub_font = load_font(44)
    body_font = load_font(38)

    y = 380
    draw.text((MARGIN, y), "Task A-MW Prototype Audit Pack", fill=(0, 0, 0), font=title_font)
    y += 120
    draw.text((MARGIN, y), "Content: Question images + prompt + options + ground truth", fill=(40, 40, 40), font=sub_font)
    y += 120

    summary = (
        f"Exam source: {exam_path}\n"
        f"Total items: {len(records)}\n"
        "Environment: MiniWorld-FourRooms-v0\n"
        "Question type: move_forward binary choice (forward feasibility)\n"
        "Page layout: Left ego, right topview (from concatenated image split)\n"
        "Suggestion: Judge based on image and prompt first, then check ground truth at bottom."
    )
    add_wrapped_block(draw, MARGIN, y, summary, body_font, PAGE_W - 2 * MARGIN, 56)
    return page


def render_question_page(record: dict[str, Any], exam_root: Path, idx: int, total: int) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(54)
    body_font = load_font(34)
    small_font = load_font(30)
    answer_font = load_font(38)

    y = MARGIN
    draw.text((MARGIN, y), f"Task A-MW Prototype Audit  {idx}/{total}", fill=(0, 0, 0), font=title_font)
    y += 86

    meta = (
        f"UID: {record['uid']}\n"
        f"Seed: {record['seed']}    Env: {record['env_task']}    Subtask: {record['subtask']}\n"
        f"Action: {record['action']['name']}    Initial direction: {en_dir(str(record.get('agent_initial_dir', '')))}"
        f" ({record.get('agent_initial_dir', '')})"
    )
    y = add_wrapped_block(draw, MARGIN, y, meta, small_font, PAGE_W - 2 * MARGIN, 42)
    y += 18

    img_path = exam_root / str(record["image_rel_path"])
    with Image.open(img_path) as im:
        concat = im.convert("RGB")
    ego_img, top_img = split_concat_vertical(concat)

    panel_gap = 40
    panel_w = (PAGE_W - 2 * MARGIN - panel_gap) // 2
    panel_h = 1120
    ego_fit = fit_image(ego_img, panel_w, panel_h)
    top_fit = fit_image(top_img, panel_w, panel_h)

    label_font = load_font(36)
    left_x = MARGIN + (panel_w - ego_fit.width) // 2
    right_panel_x = MARGIN + panel_w + panel_gap
    right_x = right_panel_x + (panel_w - top_fit.width) // 2

    draw.text((MARGIN, y), "Left: Ego (First-person view)", fill=(0, 0, 0), font=label_font)
    draw.text((right_panel_x, y), "Right: Top-view", fill=(0, 0, 0), font=label_font)
    y += 56

    page.paste(ego_fit, (left_x, y))
    page.paste(top_fit, (right_x, y))
    img_block_h = max(ego_fit.height, top_fit.height)
    y += img_block_h + 44

    prompt_title = "Question"
    draw.text((MARGIN, y), prompt_title, fill=(0, 0, 0), font=title_font)
    y += 72
    y = add_wrapped_block(draw, MARGIN, y, str(record["prompt"]), body_font, PAGE_W - 2 * MARGIN, 48)
    y += 18

    candidates_lines = ["Options:"]
    for cand in record.get("candidates", []):
        candidates_lines.append(f"{cand['label']}. {cand['value']}")
    y = add_wrapped_block(draw, MARGIN, y, "\n".join(candidates_lines), body_font, PAGE_W - 2 * MARGIN, 48)
    y += 20

    draw.rounded_rectangle(
        (MARGIN, y, PAGE_W - MARGIN, min(PAGE_H - MARGIN, y + 210)),
        radius=24,
        outline=(180, 0, 0),
        width=4,
        fill=(255, 245, 245),
    )
    y += 22
    answer_text = (
        f"Ground truth: {en_answer(str(record.get('ground_truth', '')))}\n"
        f"Position changed: {'Yes' if record.get('pos_changed') else 'No'}\n"
        f"Direction after action: {en_dir(str(record.get('after_action_dir', '')))} ({record.get('after_action_dir', '')})"
    )
    add_wrapped_block(draw, MARGIN + 24, y, answer_text, answer_font, PAGE_W - 2 * MARGIN - 48, 54, fill=(140, 0, 0))
    return page


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exam_path",
        default="tmp_miniworld/taskA-MW-prototype/taskA-MW-prototype.jsonl",
    )
    ap.add_argument(
        "--out_pdf",
        default="tmp_miniworld/taskA-MW-prototype/taskA-MW-prototype_audit_with_answers.pdf",
    )
    args = ap.parse_args()

    exam_path = Path(args.exam_path)
    records = load_records(exam_path)
    exam_root = exam_path.parent

    pages: list[Image.Image] = [build_cover(records, exam_path)]
    for idx, record in enumerate(records, start=1):
        pages.append(render_question_page(record, exam_root, idx, len(records)))

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages_rgb = [p.convert("RGB") for p in pages]
    pages_rgb[0].save(out_pdf, save_all=True, append_images=pages_rgb[1:], resolution=220.0)
    print(f"Saved audit PDF to {out_pdf}")


if __name__ == "__main__":
    main()

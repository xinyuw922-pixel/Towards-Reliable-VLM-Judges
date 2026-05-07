#!/usr/bin/env python3
"""
Renders Task A-MW v2 questions as human-reviewable PDFs.

Supports two modes:
  --mode blind        : no answer, no distractor sequence visible
  --mode answer-key   : full answer + distractor sequence visible

Usage:
  python render_taskA_MW_v2_audit_pdf.py --mode answer-key
  python render_taskA_MW_v2_audit_pdf.py --mode blind
"""
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
    raise RuntimeError(
        "No suitable font found. Please ensure msyh / NotoSansCJK / DejaVu fonts are available."
    )


def create_blank_page() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    return page, ImageDraw.Draw(page)


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> list[str]:
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


def fit_and_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    scale_w = target_w / img.width
    scale_h = target_h / img.height
    scale = max(scale_w, scale_h)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def fit_contain_on_canvas(
    img: Image.Image,
    target_w: int,
    target_h: int,
    *,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Fit an image fully inside a target box and center it on a canvas."""
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def extract_ego_panel(img: Image.Image) -> Image.Image:
    """Return the upper-half ego panel from an ego_topview_concat image."""
    return img.crop((0, 0, img.width, img.height // 2))


def overlay_topview_grid(concat_img: Image.Image, record: dict[str, Any]) -> Image.Image:
    """Overlay a scale grid on the lower-half topview panel.

    Single grid: 1 cell = 1 human-facing step unit.
    Here 1 step unit = 5 primitive MiniWorld forward steps.
    """
    scale = record.get("current_topview_scale") or {}
    try:
        x_scale = float(scale["x_scale"])
        z_scale = float(scale["z_scale"])
        x_offset = float(scale["x_offset"])
        z_offset = float(scale["z_offset"])
        forward_step = float(record["forward_step"])
    except Exception:
        return concat_img

    top_h = concat_img.height // 2
    topview = concat_img.crop((0, top_h, concat_img.width, concat_img.height)).convert("RGBA")

    cell_w = max(1.0, forward_step * 5 * x_scale)
    cell_h = max(1.0, forward_step * 5 * z_scale)

    overlay = Image.new("RGBA", topview.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    grid_color = (30, 144, 255, 80)

    def _draw_series(step_px: float, anchor_px: float, limit_px: int, vertical: bool, color, width: int) -> None:
        if step_px <= 0:
            return
        pos = anchor_px % step_px
        while pos < limit_px:
            if vertical:
                draw.line([(pos, 0), (pos, topview.height)], fill=color, width=width)
            else:
                draw.line([(0, pos), (topview.width, pos)], fill=color, width=width)
            pos += step_px

    _draw_series(cell_w, x_offset, topview.width, True, grid_color, 2)
    _draw_series(cell_h, z_offset, topview.height, False, grid_color, 2)

    merged_topview = Image.alpha_composite(topview, overlay).convert("RGB")
    out = concat_img.copy()
    out.paste(merged_topview, (0, top_h))
    return out


def en_seq(seq: list[str]) -> str:
    mapping = {
        "turn_left_30": "Turn left (1 unit)",
        "turn_right_30": "Turn right (1 unit)",
        "turn_left_60": "Turn left (2 units)",
        "turn_right_60": "Turn right (2 units)",
        "forward_5": "Forward (1 unit)",
        "forward_10": "Forward (2 units)",
    }
    return " -> ".join(mapping.get(a, a) for a in seq)


def en_prompt(record: dict[str, Any]) -> str:
    return str(record.get("prompt", "")).strip()


def en_prompt_translation(record: dict[str, Any]) -> str:
    actions = en_seq(record["action_sequence"])
    return (
        "Task: Judge the true successor state.\n"
        "You will see the agent's current state and two successor views A and B.\n"
        "This question uses the following human-review unit conventions:\n"
        "1. Forward 1 unit = 5 primitive forward steps in the original environment.\n"
        "2. Turn left/right 1 unit = 30 degrees.\n"
        "3. Turn left/right 2 units = 60 degrees.\n"
        "The topview grid aligns with these distance units: 1 grid cell = 1 unit.\n"
        f"Please execute the action sequence in order from the current state: [{actions}].\n"
        "No need to convert to real meters; judge based on relative displacement and orientation changes.\n"
        "Question: After fully executing this action sequence, which of candidate A or candidate B is the true successor state?\n"
        "Answer requirement: Output only a single letter A or B."
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def infer_exam_label(exam_path: Path) -> str:
    stem = exam_path.stem
    return stem


def infer_env_label(records: list[dict[str, Any]]) -> str:
    if not records:
        return "unknown"
    env_task = str(records[0].get("env_task", "unknown"))
    mapping = {
        "fourrooms": "MiniWorld-FourRooms-v0",
        "pickupobjects": "MiniWorld-PickupObjects-v0",
        "putnext": "MiniWorld-PutNext-v0",
    }
    return mapping.get(env_task, env_task)


def build_cover(
    records: list[dict[str, Any]], exam_path: Path, mode: str
) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(44)
    body_font = load_font(38)

    y = 280
    mode_label = "(Blind Review)" if mode == "blind" else "(Answer Key Version)"
    exam_label = infer_exam_label(exam_path)
    env_label = infer_env_label(records)
    draw.text(
        (MARGIN, y),
        f"Task A-MW v2 Audit Pack {mode_label}",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 110
    draw.text(
        (MARGIN, y),
        "Question Type: Successor State Choice",
        fill=(0, 0, 0),
        font=sub_font,
    )
    y += 90

    content_note = (
        "Page content: Current state (ego+topview) + Candidate A (ego only) + Candidate B (ego only) + Action sequence"
        if mode == "blind"
        else "Page content: Current state (ego+topview) + Candidate A (ego only) + Candidate B (ego only) + Action sequence + Correct answer + Distractor sequence"
    )
    draw.text((MARGIN, y), content_note, fill=(40, 40, 40), font=sub_font)
    y += 120

    seq_lengths: dict[int, int] = {}
    strat_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for r in records:
        n = len(r["action_sequence"])
        seq_lengths[n] = seq_lengths.get(n, 0) + 1
        strat = r.get("distractor_strategy", "unknown")
        strat_counts[strat] = strat_counts.get(strat, 0) + 1
        bucket = r.get("difficulty_bucket", "unknown")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    a_count = sum(1 for r in records if r["ground_truth"] == "A")
    b_count = sum(1 for r in records if r["ground_truth"] == "B")

    summary_lines = [
        f"Exam source: {exam_path}",
        f"Exam label: {exam_label}",
        f"Total items: {len(records)} (A={a_count} / B={b_count})",
        f"Environment: {env_label}",
        "Current state: ego_topview_concat (top=ego first-person, bottom=topview)",
        "Candidate successor: Ego first-person view only, no candidate topview.",
        "Question text: English version provided for each item.",
        "Human-review unit: Forward 1 unit = 5 primitive forward steps; Turn left/right 1 unit = 30 degrees; Turn left/right 2 units = 60 degrees.",
        "Topview grid: 1 cell = 1 unit forward.",
        f"Difficulty distribution: {dict(sorted(bucket_counts.items()))} (easy = current room contains red block; hard = current room has no red block)",
        f"Distractor strategy distribution: {dict(sorted(strat_counts.items()))}",
        f"Action sequence length distribution: {dict(sorted(seq_lengths.items()))}",
        (
            "Blind review: Please judge which of A/B is more likely the true successor without looking at the answer."
            if mode == "blind"
            else "Answer key version: Contains correct answer and distractor sequence, for verification only."
        ),
    ]
    y = add_wrapped_block(
        draw, MARGIN, y, "\n".join(summary_lines), body_font, PAGE_W - 2 * MARGIN, 55
    )
    return page


def render_question_page(
    record: dict[str, Any], exam_root: Path, idx: int, total: int, mode: str
) -> Image.Image:
    """Render one question page. In blind mode, omits answer box and distractor sequence."""
    page, draw = create_blank_page()
    title_font = load_font(46)
    label_font = load_font(30)
    small_font = load_font(30)
    body_font = load_font(34)
    prompt_font = load_font(28)
    answer_font = load_font(38)

    y = MARGIN
    mode_tag = "[Blind]" if mode == "blind" else "[Answer Key]"
    draw.text(
        (MARGIN, y),
        f"Task A-MW v2  {mode_tag}  {idx}/{total}",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 80

    meta = (
        f"UID: {record['uid']}    Seed: {record['seed']}    Env: {record['env_task']}"
        f"    Difficulty: {record.get('difficulty_bucket', 'unknown')}"
        f"    Distractor Strategy: {record.get('distractor_strategy', 'unknown')}"
    )
    draw.text((MARGIN, y), meta, fill=(60, 60, 60), font=small_font)
    y += 44

    # Load all three images
    current_path = exam_root / record["current_image_rel_path"]
    cand_a_path = exam_root / record["candidate_a_rel_path"]
    cand_b_path = exam_root / record["candidate_b_rel_path"]

    with Image.open(current_path) as im:
        current_img = im.convert("RGB")
    with Image.open(cand_a_path) as im:
        cand_a_img = im.convert("RGB")
    with Image.open(cand_b_path) as im:
        cand_b_img = im.convert("RGB")

    current_img = overlay_topview_grid(current_img, record)
    cand_a_ego = extract_ego_panel(cand_a_img)
    cand_b_ego = extract_ego_panel(cand_b_img)

    img_gap = 50
    content_w = PAGE_W - 2 * MARGIN

    # Current image at top (full width)
    current_panel_h = 820
    current_fit = fit_contain_on_canvas(current_img, content_w, current_panel_h)
    page.paste(current_fit, (MARGIN, y))
    draw.text(
        (MARGIN, y + current_fit.height + 12),
        "Current State (Before executing action sequence)",
        fill=(60, 60, 60),
        font=label_font,
    )
    y += current_fit.height + 42

    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=(200, 200, 200), width=2)
    y += 30

    # A/B candidates side by side
    ab_panel_w = (content_w - img_gap) // 2
    ab_panel_h = 520

    cand_a_fit = fit_and_crop_to_aspect(cand_a_ego, ab_panel_w, ab_panel_h)
    cand_b_fit = fit_and_crop_to_aspect(cand_b_ego, ab_panel_w, ab_panel_h)

    cand_a_x = MARGIN
    cand_b_x = MARGIN + ab_panel_w + img_gap

    draw.text(
        (cand_a_x, y),
        "Candidate A (Ego successor only)",
        fill=(0, 0, 0),
        font=label_font,
    )
    draw.text(
        (cand_b_x, y),
        "Candidate B (Ego successor only)",
        fill=(0, 0, 0),
        font=label_font,
    )
    y += 44

    page.paste(cand_a_fit, (cand_a_x, y))
    page.paste(cand_b_fit, (cand_b_x, y))
    y += ab_panel_h + 30

    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=(200, 200, 200), width=2)
    y += 30

    # Action sequences
    seq_font = load_font(34)
    draw.text(
        (MARGIN, y),
        "Action Sequence (Which candidate matches the true result after executing these actions?)",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 60

    correct_seq_str = "Correct sequence: " + en_seq(record["action_sequence"])
    draw.text(
        (MARGIN, y), correct_seq_str, fill=(0, 80, 160), font=seq_font
    )
    y += 52

    # Only show distractor sequence in answer-key mode
    if mode == "answer-key":
        dist_seq_str = "Distractor sequence: " + en_seq(record["distractor_sequence"])
        draw.text(
            (MARGIN, y), dist_seq_str, fill=(120, 120, 120), font=seq_font
        )
        y += 52

    y += 20

    # Prompt
    draw.text((MARGIN, y), "Question (English):", fill=(0, 0, 0), font=body_font)
    y += 40
    y = add_wrapped_block(
        draw, MARGIN, y, en_prompt(record), prompt_font, content_w, 38
    )
    y += 18
    draw.text((MARGIN, y), "Question (Chinese Translation):", fill=(0, 0, 0), font=body_font)
    y += 40
    y = add_wrapped_block(
        draw, MARGIN, y, en_prompt_translation(record), prompt_font, content_w, 38
    )
    y += 20

    # Answer box — only in answer-key mode
    if mode == "answer-key":
        answer_box_top = min(y, PAGE_H - MARGIN - 210)
        draw.rounded_rectangle(
            (
                MARGIN,
                answer_box_top,
                PAGE_W - MARGIN,
                min(PAGE_H - MARGIN, answer_box_top + 180),
            ),
            radius=24,
            outline=(180, 0, 0),
            width=4,
            fill=(255, 245, 245),
        )
        y = answer_box_top + 24
        gt = record["ground_truth"]
        cc = record["correct_candidate"]
        answer_text = (
            f"Ground truth: {gt} (Correct candidate = {cc})\n"
            f"Correct candidate position: {record['correct_candidate_agent_pos']}  Direction: {record['correct_candidate_agent_cardinal']}\n"
            f"Distractor candidate position: {record['distractor_candidate_agent_pos']}  Direction: {record['distractor_candidate_agent_cardinal']}\n"
            f"Current state position: {record['current_agent_pos']}  Direction: {record['current_agent_cardinal']}\n"
            f"Distractor source: {record['distractor_source']}  Strategy: {record['distractor_strategy']}"
        )
        add_wrapped_block(
            draw,
            MARGIN + 24,
            y,
            answer_text,
            answer_font,
            content_w - 48,
            54,
            fill=(140, 0, 0),
        )

    return page


def render_pdf(exam_path: Path, out_pdf: Path, mode: str) -> None:
    """Render all questions as a PDF in the specified mode."""
    records = load_records(exam_path)
    exam_root = exam_path.parent

    pages: list[Image.Image] = [build_cover(records, exam_path, mode)]
    for idx, record in enumerate(records, start=1):
        pages.append(
            render_question_page(record, exam_root, idx, len(records), mode)
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages_rgb = [p.convert("RGB") for p in pages]
    pages_rgb[0].save(
        out_pdf,
        save_all=True,
        append_images=pages_rgb[1:],
        resolution=220.0,
    )
    print(f"Saved {mode} PDF to {out_pdf}  ({len(pages)} pages)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task A-MW v2 audit PDF renderer")
    ap.add_argument(
        "--exam_path",
        default="tmp_miniworld/taskA-MW-v2-prototype/taskA-MW-v2-prototype.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="tmp_miniworld/taskA-MW-v2-prototype",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="answer-key",
        help="'blind' = no answer; 'answer-key' = full answer (default: answer-key)",
    )
    args = ap.parse_args()

    exam_path = Path(args.exam_path)
    out_dir = Path(args.out_dir)
    exam_label = infer_exam_label(exam_path)

    mode_str = "blind" if args.mode == "blind" else "audit_with_answers"
    out_pdf = out_dir / f"{exam_label}_{mode_str}.pdf"
    render_pdf(exam_path, out_pdf, args.mode)

    # If no explicit mode specified, also generate the other one
    if args.mode != "blind":
        blind_out = out_dir / f"{exam_label}_blind.pdf"
        render_pdf(exam_path, blind_out, "blind")
    if args.mode != "answer-key":
        ak_out = out_dir / f"{exam_label}_audit_with_answers.pdf"
        render_pdf(exam_path, ak_out, "answer-key")


if __name__ == "__main__":
    main()

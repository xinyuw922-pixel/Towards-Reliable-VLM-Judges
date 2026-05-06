#!/usr/bin/env python3
"""
D-MW Trajectory Judgment PDF Audit Renderer.

Produces two PDF variants for a D-MW FourRooms exam:
  --mode blind       : no answer, no success/failure sequence visible
  --mode answer-key  : full answer + success/failure sequences + terminal state metadata

Usage:
  python render_taskD_MW_audit_pdf.py --mode blind
  python render_taskD_MW_audit_pdf.py --mode answer-key
"""
from __future__ import annotations

import argparse
import json
import math
import re
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
STEPS_PER_UNIT = 10


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    raise RuntimeError(
        "未找到可用的中文字体，无法生成中文审计 PDF。"
        "请确认系统中存在 msyh / NotoSansCJK / 文泉驿 等中文字体。"
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


def fit_contain_on_canvas(
    img: Image.Image,
    target_w: int,
    target_h: int,
    *,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Topview grid overlay — 1 cell = 1 unit = STEPS_PER_UNIT primitive forward steps
# ---------------------------------------------------------------------------

def overlay_topview_grid(concat_img: Image.Image, record: dict[str, Any]) -> Image.Image:
    """Overlay a scale grid on the lower-half topview panel.

    Grid convention: 1 cell = 1 unit = STEPS_PER_UNIT primitive forward steps.
    In FourRooms: forward_step=0.15, STEPS_PER_UNIT=10,
    so 1 cell = 0.15 * 10 = 1.5 grid units.
    """
    scale_info = record.get("current_topview_scale") or record.get("topview_scale") or {}
    try:
        x_scale = float(scale_info["x_scale"])
        z_scale = float(scale_info["z_scale"])
        x_offset = float(scale_info["x_offset"])
        z_offset = float(scale_info["z_offset"])
        forward_step = float(record["forward_step"])
    except Exception:
        return concat_img

    steps_per_unit = record.get("steps_per_unit", STEPS_PER_UNIT)
    unit_grid_size = forward_step * steps_per_unit

    top_h = concat_img.height // 2
    topview = concat_img.crop((0, top_h, concat_img.width, concat_img.height)).convert("RGBA")

    cell_w = max(1.0, unit_grid_size * x_scale)
    cell_h = max(1.0, unit_grid_size * z_scale)

    overlay = Image.new("RGBA", topview.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    grid_color = (30, 144, 255, 80)

    def _draw_series(
        step_px: float, anchor_px: float, limit_px: int, vertical: bool, color, width: int
    ) -> None:
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


# ---------------------------------------------------------------------------
# English / Chinese prompt helpers
# ---------------------------------------------------------------------------

def en_prompt(record: dict[str, Any]) -> str:
    planned = record.get("planned_trajectory", "")
    forward_step = record.get("forward_step", 0.15)
    turn_step_deg = record.get("turn_step_deg", 15.0)
    steps_per_unit = record.get("steps_per_unit", STEPS_PER_UNIT)
    unit_grid_size = round(forward_step * steps_per_unit, 2)

    turn_note = ""
    if "turn" in planned.lower():
        m = re.search(r"(\d+(?:\.\d+)?)\s*°", planned)
        if m:
            total_angle = float(m.group(1))
            turn_note = (
                f"\n(Each primitive turn step = {turn_step_deg:.1f}°. "
                f"Turning therefore totals {total_angle:.1f}°.)"
            )

    return (
        "Task: Trajectory Judgment\n"
        "Judge whether the trajectory outcome is successful or failed.\n"
        "\n"
        "You are given an initial state view (ego view + topview) and two candidate outcome views.\n"
        f"Starting from the initial state, the following actions were executed:\n"
        f"  [{planned}]\n"
        "Action unit convention:\n"
        f"  - 1 unit = {steps_per_unit} primitive forward steps\n"
        f"  - 1 unit = {forward_step:.2f} × {steps_per_unit} = ~{unit_grid_size:.2f} grid units\n"
        f"  - Each primitive forward step = {forward_step:.2f} grid units\n"
        f"  - The topview grid is labeled in these units (1 cell = 1 unit).{turn_note}\n"
        "One of the two outcome views shows the actual result:\n"
        "  - Successful: the agent clearly travels through the environment "
        "(through doors, corridors, or a long distance).\n"
        "  - Failed: the agent hits a wall and remains stuck near the starting position.\n"
        "The trajectory is successful if the agent clearly travels; "
        "it fails if the agent remains stuck near the starting position.\n"
        "\n"
        "Which outcome view, A or B, matches the actual result?\n"
        "Answer with a single letter: A or B."
    )


def zh_prompt(record: dict[str, Any]) -> str:
    planned = record.get("planned_trajectory", "")
    forward_step = record.get("forward_step", 0.15)
    steps_per_unit = record.get("steps_per_unit", STEPS_PER_UNIT)
    unit_grid_size = round(forward_step * steps_per_unit, 2)
    return (
        "题型：轨迹结果判断（Trajectory Judgment）\n"
        f"从初始状态出发，执行了以下动作：[{planned}]\n"
        "动作单位约定：\n"
        f"  - 1 unit = {steps_per_unit} 步 primitive 前进步\n"
        f"  - 1 unit = {forward_step:.2f} × {steps_per_unit} ≈ {unit_grid_size:.2f} 个网格单位\n"
        f"  - 每次 primitive 前进一步移动 {forward_step:.2f} 个网格单位\n"
        "  - 俯视图网格按 unit 标注（1格 = 1 unit）\n"
        "两个候选结果图中，一张显示代理成功穿过了环境（穿过门，大幅位移），"
        "另一张显示代理撞墙后停在起始位置附近。\n"
        "轨迹成功的判定标准：代理明显在环境中穿行。"
        "轨迹失败的判定标准：代理困在起始位置附近。\n"
        "哪张候选图匹配这次轨迹的真实结果？\n"
        "作答要求：只回答一个字母 A 或 B。"
    )


def zh_seq(seq: list[str]) -> str:
    mapping = {
        "turn_left": "左转(primitive)",
        "turn_right": "右转(primitive)",
        "move_forward": "前进(primitive)",
    }
    return " → ".join(mapping.get(a, a) for a in seq)


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def build_cover(
    records: list[dict[str, Any]], exam_path: Path, mode: str
) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(44)
    body_font = load_font(38)

    y = 280
    mode_label = "（盲审版 — 不含答案）" if mode == "blind" else "（答案版 — 含标准答案）"
    draw.text(
        (MARGIN, y),
        f"Trajectory Judgment D-MW 审计包 {mode_label}",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 110
    draw.text(
        (MARGIN, y),
        "题型：轨迹结果判断（Trajectory Judgment）",
        fill=(0, 0, 0),
        font=sub_font,
    )
    y += 90

    content_note = (
        "每页内容：初始状态图（ego+俯视图，已标注栅格） + "
        "候选A图（含完整ego+俯视，已标注栅格） + "
        "候选B图（含完整ego+俯视，已标注栅格） + 轨迹描述"
        if mode == "blind"
        else (
            "每页内容：初始状态图（ego+俯视图，已标注栅格） + "
            "候选A图（含完整ego+俯视，已标注栅格） + "
            "候选B图（含完整ego+俯视，已标注栅格） + "
            "轨迹描述 + 正确答案 + success/failure 终态元数据"
        )
    )
    draw.text((MARGIN, y), content_note, fill=(40, 40, 40), font=sub_font)
    y += 120

    a_count = sum(1 for r in records if r["ground_truth"] == "A")
    b_count = sum(1 for r in records if r["ground_truth"] == "B")

    forward_step = records[0].get("forward_step", 0.15) if records else 0.15
    steps_per_unit = records[0].get("steps_per_unit", STEPS_PER_UNIT) if records else STEPS_PER_UNIT
    unit_grid = round(forward_step * steps_per_unit, 2)

    success_dirs: dict[str, int] = {}
    failure_dirs: dict[str, int] = {}
    planned_kinds: dict[str, int] = {}
    for r in records:
        sd = r.get("success_dir", "?")
        fd = r.get("failure_dir", "?")
        pt = r.get("planned_trajectory", "?")
        success_dirs[sd] = success_dirs.get(sd, 0) + 1
        failure_dirs[fd] = failure_dirs.get(fd, 0) + 1
        if "turn" not in pt.lower():
            planned_kinds["直行"] = planned_kinds.get("直行", 0) + 1
        else:
            planned_kinds["有转向"] = planned_kinds.get("有转向", 0) + 1

    summary_lines = [
        f"题目来源：{exam_path}",
        f"总题数：{len(records)}（A={a_count} / B={b_count}）",
        "环境：MiniWorld-FourRooms-v0",
        "当前状态图：ego_topview_concat（上=ego第一人称，下=topview俯视图，已标注栅格）",
        "俯视栅格约定：1 格 = 1 unit = 10 步 primitive 前进步 ≈ "
        f"{unit_grid:.2f} 个网格单位",
        "题型说明：给定一个「沿某方向前进若干 unit」的规划轨迹，判断代理最终是"
        "成功穿行（穿过门，大幅位移）还是撞墙失败（困在起点附近）。",
        "成功语义：代理有明显位移（穿过门或经过长走廊）。",
        "失败语义：代理撞墙，未能离开起始区域（停在起点附近）。",
        f"规划轨迹分布（按方向）：{dict(sorted(success_dirs.items()))}",
        f"失败方向分布：{dict(sorted(failure_dirs.items()))}",
        f"轨迹类型：{dict(sorted(planned_kinds.items()))}",
        (
            "盲审说明：请先不看答案，根据初始图+轨迹描述，判断A/B哪个是真实轨迹结果。"
            if mode == "blind"
            else "答案版说明：含标准答案和终态位置/朝向元数据，仅供复核使用。"
        ),
    ]
    y = add_wrapped_block(
        draw, MARGIN, y, "\n".join(summary_lines), body_font, PAGE_W - 2 * MARGIN, 55
    )
    return page


# ---------------------------------------------------------------------------
# Question page
# ---------------------------------------------------------------------------

def render_question_page(
    record: dict[str, Any], exam_root: Path, idx: int, total: int, mode: str
) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(46)
    label_font = load_font(30)
    small_font = load_font(30)
    body_font = load_font(34)
    prompt_font = load_font(28)
    answer_font = load_font(36)

    content_w = PAGE_W - 2 * MARGIN
    y = MARGIN

    mode_tag = "[盲审]" if mode == "blind" else "[答案版]"
    draw.text(
        (MARGIN, y),
        f"Trajectory Judgment  轨迹结果判断  {mode_tag}  {idx}/{total}",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 80

    meta_parts = [
        f"UID: {record['uid']}",
        f"Seed: {record['seed']}",
        f"环境: {record['env_task']}",
        f"成功方向: {record.get('success_dir', '?')}",
        f"失败方向: {record.get('failure_dir', '?')}",
    ]
    draw.text((MARGIN, y), "    ".join(meta_parts), fill=(60, 60, 60), font=small_font)
    y += 44

    # Load all three images
    initial_path = exam_root / record["initial_image_rel_path"]
    cand_a_path = exam_root / record["candidate_a_rel_path"]
    cand_b_path = exam_root / record["candidate_b_rel_path"]

    with Image.open(initial_path) as im:
        initial_img = im.convert("RGB")
    with Image.open(cand_a_path) as im:
        cand_a_img = im.convert("RGB")
    with Image.open(cand_b_path) as im:
        cand_b_img = im.convert("RGB")

    # Overlay topview grid on ALL three images (initial + candidates)
    initial_img_grid = overlay_topview_grid(initial_img, record)
    cand_a_img_grid = overlay_topview_grid(cand_a_img, record)
    cand_b_img_grid = overlay_topview_grid(cand_b_img, record)

    img_gap = 50

    # Initial state (full width, ego+topview with grid)
    initial_panel_h = 680
    initial_fit = fit_contain_on_canvas(initial_img_grid, content_w, initial_panel_h)
    page.paste(initial_fit, (MARGIN, y))
    draw.text(
        (MARGIN, y + initial_fit.height + 12),
        "初始状态图（Initial State — 执行轨迹前的状态，已标注栅格）",
        fill=(60, 60, 60),
        font=label_font,
    )
    y += initial_fit.height + 44

    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=(200, 200, 200), width=2)
    y += 30

    # A/B candidates — full images with grid, side by side
    ab_panel_w = (content_w - img_gap) // 2
    ab_panel_h = 600

    cand_a_fit = fit_contain_on_canvas(cand_a_img_grid, ab_panel_w, ab_panel_h)
    cand_b_fit = fit_contain_on_canvas(cand_b_img_grid, ab_panel_w, ab_panel_h)

    cand_a_x = MARGIN
    cand_b_x = MARGIN + ab_panel_w + img_gap

    draw.text(
        (cand_a_x, y),
        "候选 A（候选结果图 A，已标注栅格）",
        fill=(0, 0, 0),
        font=label_font,
    )
    draw.text(
        (cand_b_x, y),
        "候选 B（候选结果图 B，已标注栅格）",
        fill=(0, 0, 0),
        font=label_font,
    )
    y += 44

    page.paste(cand_a_fit, (cand_a_x, y))
    page.paste(cand_b_fit, (cand_b_x, y))
    y += ab_panel_h + 30

    draw.line([(MARGIN, y), (PAGE_W - MARGIN, y)], fill=(200, 200, 200), width=2)
    y += 30

    # Planned trajectory
    planned = record.get("planned_trajectory", "?")
    draw.text(
        (MARGIN, y),
        "规划轨迹（Planned Trajectory — 从初始状态执行的动作序列）：",
        fill=(0, 0, 0),
        font=title_font,
    )
    y += 60

    if mode == "answer-key":
        draw.text(
            (MARGIN, y),
            f"规划轨迹：{planned}",
            fill=(0, 80, 160),
            font=load_font(34),
        )
        y += 52
        s_seq = record.get("success_sequence", [])
        f_seq = record.get("failure_sequence", [])
        draw.text(
            (MARGIN, y),
            f"success 序列（primitive）：{zh_seq(s_seq)}",
            fill=(0, 100, 0),
            font=load_font(30),
        )
        y += 44
        draw.text(
            (MARGIN, y),
            f"failure 序列（primitive）：{zh_seq(f_seq)}",
            fill=(100, 100, 100),
            font=load_font(30),
        )
        y += 44
    else:
        draw.text(
            (MARGIN, y),
            f"规划轨迹：{planned}",
            fill=(0, 0, 0),
            font=load_font(34),
        )
        y += 52

    y += 20

    # Prompt text
    draw.text((MARGIN, y), "Question (English):", fill=(0, 0, 0), font=body_font)
    y += 40
    y = add_wrapped_block(
        draw, MARGIN, y, en_prompt(record), prompt_font, content_w, 38
    )
    y += 18
    draw.text((MARGIN, y), "中文翻译：", fill=(0, 0, 0), font=body_font)
    y += 40
    y = add_wrapped_block(
        draw, MARGIN, y, zh_prompt(record), prompt_font, content_w, 38
    )
    y += 20

    # Answer box — only in answer-key mode
    if mode == "answer-key":
        answer_box_top = min(y + 10, PAGE_H - MARGIN - 210)
        draw.rounded_rectangle(
            (
                MARGIN,
                answer_box_top,
                PAGE_W - MARGIN,
                min(PAGE_H - MARGIN, answer_box_top + 195),
            ),
            radius=24,
            outline=(180, 0, 0),
            width=4,
            fill=(255, 245, 245),
        )
        y = answer_box_top + 24
        gt = record["ground_truth"]
        initial_pos = record.get("initial_agent_pos", [])
        success_pos = record.get("success_agent_pos", [])
        failure_pos = record.get("failure_agent_pos", [])

        success_dist = 0.0
        failure_dist = 0.0
        if initial_pos and success_pos:
            success_dist = math.sqrt(
                (success_pos[0] - initial_pos[0]) ** 2
                + (success_pos[1] - initial_pos[1]) ** 2
            )
        if initial_pos and failure_pos:
            failure_dist = math.sqrt(
                (failure_pos[0] - initial_pos[0]) ** 2
                + (failure_pos[1] - initial_pos[1]) ** 2
            )

        answer_text = (
            f"标准答案：{gt}（正确答案 = 候选 {gt}）\n"
            f"初始状态：位置{initial_pos}  朝向{record.get('initial_agent_cardinal', '?')}\n"
            f"Success 终态：位置{success_pos}  朝向{record.get('success_agent_cardinal', '?')}  "
            f"位移距离≈{success_dist:.3f}\n"
            f"Failure 终态：位置{failure_pos}  朝向{record.get('failure_agent_cardinal', '?')}  "
            f"位移距离≈{failure_dist:.3f}\n"
            f"失败是否留在起点附近：{record.get('failure_stayed_near_start', '?')}  "
            f"失败位移≈{record.get('failure_travel_distance', 0):.3f}\n"
            f"成功方向：{record.get('success_dir', '?')}  "
            f"失败方向：{record.get('failure_dir', '?')}"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_pdf(exam_path: Path, out_pdf: Path, mode: str) -> None:
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
    ap = argparse.ArgumentParser(description="D-MW Trajectory Judgment Audit PDF Renderer")
    ap.add_argument(
        "--jsonl",
        default="tmp_miniworld/taskA-MW-D-fourrooms/taskA-MW-D-fourrooms.jsonl",
        help="Path to D-MW exam JSONL",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="blind",
        help="'blind' = no answers; 'answer-key' = with answers and terminal state metadata",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output PDF path.",
    )
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_pdf = Path(args.out) if args.out else (
        jsonl_path.parent / f"D-MW_fourrooms_{args.mode}_audit.pdf"
    )
    render_pdf(jsonl_path, out_pdf, args.mode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

FAMILY_ZH = {
    "doorkey": "DoorKey / 钥匙开门",
    "multiroom": "MultiRoom / 多房间",
    "redblue": "RedBlue / 红蓝门",
}

DIFFICULTY_ZH = {
    "structural": "结构增强",
    "structural_v2": "结构增强 v2",
}

INTERVENTION_ZH = {
    "remove_key": "移除钥匙",
    "lock_gate2_forever": "第二扇门永久锁定",
}


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


def add_wrapped_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    line_gap: int,
    *,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> int:
    for line in wrap_text(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += line_gap
    return y


def fit_contain(
    img: Image.Image,
    target_w: int,
    target_h: int,
    *,
    bg: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def zh_family(family: str) -> str:
    return FAMILY_ZH.get(family, family)


def zh_difficulty(difficulty: str) -> str:
    return DIFFICULTY_ZH.get(difficulty, difficulty)


def load_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    family_rank = {"doorkey": 0, "multiroom": 1, "redblue": 2}
    diff_rank = {"structural": 0, "structural_v2": 1}
    rows.sort(
        key=lambda r: (
            family_rank.get(r["family"], 9),
            diff_rank.get(str(r["layout_metadata"].get("difficulty", "")), 9),
        )
    )
    return rows


def load_variant_manifest(root: Path, row: dict[str, Any], variant: str) -> dict[str, Any]:
    folder = root / row["family"] / row["template_id"]
    path = folder / f"storyboard_{variant}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_cover(rows: list[dict[str, Any]], mode: str, summary_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(34)

    draw.text((MARGIN, 120), "MiniWorld Task D 结构增强试卷", font=title_font, fill=(0, 0, 0))
    mode_label = "盲审版" if mode == "blind" else "答案版"
    draw.text((MARGIN, 230), mode_label, font=sub_font, fill=(0, 0, 0))

    lines = [
        f"数据索引：{summary_path}",
        f"题目总数：{len(rows)}",
        "",
        "本试卷仅展示结构增强版本：",
        "在保持事件规则不变的前提下，提高房间拓扑复杂度，并加入额外转角。",
        "",
        "本版采用逐帧 square montage 排版，尽量避免细长 storyboard 在页面中被压缩。",
        "",
        "盲审版：",
        "只展示 full 轨迹的方阵式连续帧，不显示答案。",
        "",
        "答案版：",
        "展示 full / cf 的方阵式连续帧、关键事件覆盖图和事件步信息。",
    ]
    add_wrapped_text(draw, MARGIN, 340, "\n".join(lines), body_font, PAGE_W - 2 * MARGIN, 48)

    y = 960
    for row in rows:
        line = f"{zh_family(row['family'])} / {zh_difficulty(str(row['layout_metadata'].get('difficulty', '')))} / {row['template_id']}"
        y = add_wrapped_text(draw, MARGIN, y, line, body_font, PAGE_W - 2 * MARGIN, 44)
        y += 10
    return page


def build_prompt_block(row: dict[str, Any], *, blind: bool) -> str:
    family = zh_family(row["family"])
    difficulty = zh_difficulty(str(row["layout_metadata"].get("difficulty", "unknown")))
    note = row.get("note", "")
    lines = [
        f"环境家族：{family}",
        f"难度：{difficulty}",
        f"模板名：{row['template_id']}",
        f"说明：{note}",
        "",
        "题目：判断这条连续帧轨迹是否成功完成任务。",
        "作答要求：只输出一个英文单词：SUCCESS 或 FAIL。",
    ]
    if not blind:
        lines.extend(
            [
                "",
                f"配对是否合格：{row['pair_ok']}",
                f"cf 干预方式：{INTERVENTION_ZH.get(row['cf_intervention'], row['cf_intervention'])}",
                f"full 事件步：{row['full_event_steps']}",
                f"cf 事件步：{row['cf_event_steps']}",
            ]
        )
    return "\n".join(lines)


def expected_frame_paths(manifest: dict[str, Any], base_dir: Path) -> list[tuple[int, str, Path]]:
    frames_dir = base_dir / manifest["frames_dir_relpath"]
    out: list[tuple[int, str, Path]] = []
    for rec in manifest["step_records"]:
        t = int(rec["t"])
        action = rec["action"] or "start"
        exact = frames_dir / f"t{t:02d}_{action}.png"
        if exact.exists():
            out.append((t, action, exact))
            continue
        alts = sorted(frames_dir.glob(f"t{t:02d}_*.png"))
        if alts:
            out.append((t, action, alts[0]))
    return out


def build_square_montage(manifest: dict[str, Any], base_dir: Path, *, target_side: int) -> Image.Image:
    items = expected_frame_paths(manifest, base_dir)
    cols = max(4, math.ceil(math.sqrt(len(items))))
    rows = math.ceil(len(items) / cols)
    gap = 14
    cell_side = 220
    label_h = 28
    canvas = Image.new(
        "RGB",
        (cols * cell_side + (cols + 1) * gap, rows * cell_side + (rows + 1) * gap),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(18)
    for idx, (t, action, path) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = gap + col * (cell_side + gap)
        y = gap + row * (cell_side + gap)
        draw.rounded_rectangle((x, y, x + cell_side, y + cell_side), radius=14, fill=(255, 255, 255), outline=(210, 210, 210), width=2)
        frame = Image.open(path).convert("RGB")
        fitted = fit_contain(frame, cell_side - 12, cell_side - label_h - 14, bg=(255, 255, 255))
        canvas.paste(fitted, (x + 6, y + 6))
        label = f"t{t:02d} {action}"
        text_w = draw.textlength(label, font=label_font)
        draw.text((x + (cell_side - text_w) / 2, y + cell_side - label_h), label, font=label_font, fill=(70, 70, 70))
    return fit_contain(canvas, target_side, target_side, bg=(255, 255, 255))


def build_row_page(row: dict[str, Any], *, root: Path, mode: str) -> Image.Image:
    blind = mode == "blind"
    page, draw = create_blank_page()
    title_font = load_font(52)
    body_font = load_font(30)
    small_font = load_font(24)

    title = f"{zh_family(row['family'])} / {zh_difficulty(str(row['layout_metadata'].get('difficulty', '')))} / {row['template_id']}"
    draw.text((MARGIN, 70), title, font=title_font, fill=(0, 0, 0))

    prompt_y = 150
    prompt_h = 390 if blind else 440
    draw.rounded_rectangle((MARGIN, prompt_y, PAGE_W - MARGIN, prompt_y + prompt_h), radius=24, outline=(0, 0, 0), width=3)
    add_wrapped_text(
        draw,
        MARGIN + 24,
        prompt_y + 24,
        build_prompt_block(row, blind=blind),
        body_font,
        PAGE_W - 2 * MARGIN - 48,
        40,
    )

    full_manifest = load_variant_manifest(root, row, "full")
    full_dir = (root / row["family"] / row["template_id"]).resolve()
    full_montage = build_square_montage(full_manifest, full_dir, target_side=1600 if blind else 1120)

    if blind:
        page.paste(full_montage, ((PAGE_W - full_montage.width) // 2, 620))
        draw.text((MARGIN, 560), "Full 连续帧方阵图", font=body_font, fill=(0, 0, 0))
        return page

    cf_manifest = load_variant_manifest(root, row, "cf")
    cf_dir = (root / row["family"] / row["template_id"]).resolve()
    cf_montage = build_square_montage(cf_manifest, cf_dir, target_side=660)

    event_img = Image.open(root / row["event_coverage_relpath"]).convert("RGB")
    event_fit = fit_contain(event_img, 660, 660, bg=(255, 255, 255))

    left_x = MARGIN
    top_y = 650
    page.paste(full_montage, (left_x, top_y))
    draw.text((left_x, 602), "Full 连续帧方阵图", font=body_font, fill=(0, 0, 0))

    right_x = PAGE_W - MARGIN - 660
    page.paste(cf_montage, (right_x, top_y))
    draw.text((right_x, 602), "CF 连续帧方阵图", font=body_font, fill=(0, 0, 0))

    event_y = top_y + 760
    page.paste(event_fit, (right_x, event_y))
    draw.text((right_x, event_y - 46), "关键事件覆盖图", font=body_font, fill=(0, 0, 0))

    meta_lines = [
        f"layout = {row['layout_metadata'].get('world_variant')}",
        f"route = {row['layout_metadata'].get('route_variant')}",
        f"full 事件步 = {row['full_event_steps']}",
        f"cf 事件步 = {row['cf_event_steps']}",
    ]
    add_wrapped_text(draw, left_x, 1840, "\n".join(meta_lines), small_font, 1120, 34, fill=(70, 70, 70))
    return page


def save_pdf(pages: list[Image.Image], out_pdf: Path) -> None:
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(out_pdf, save_all=True, append_images=rgb_pages[1:], resolution=150.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render structural MiniWorld Task D PDFs with square montages")
    ap.add_argument(
        "--summary_json",
        default="tmp_miniworld/taskD-MW-structural-showcase-v2/structural_showcase_summary.json",
    )
    ap.add_argument("--mode", choices=["blind", "answer-key"], default="answer-key")
    ap.add_argument("--out_pdf", default=None)
    args = ap.parse_args()

    summary_path = Path(args.summary_json)
    root = summary_path.parent
    rows = load_summary(summary_path)

    pages = [build_cover(rows, args.mode, summary_path)]
    for row in rows:
        pages.append(build_row_page(row, root=root, mode=args.mode))

    if args.out_pdf is None:
        suffix = "blind_zh" if args.mode == "blind" else "answer_key_zh"
        out_pdf = root / f"taskD_MW_structural_showcase_{suffix}.pdf"
    else:
        out_pdf = Path(args.out_pdf)
    save_pdf(pages, out_pdf)
    print(f"Saved PDF to {out_pdf}")


if __name__ == "__main__":
    main()

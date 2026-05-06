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

PAGE_W = 2000
PAGE_H = 2828
MARGIN = 90

FAMILY_ZH = {
    "doorkey": "DoorKey / 钥匙开门",
    "multiroom": "MultiRoom / 多房间",
    "redblue": "RedBlue / 红蓝门",
}

DIFFICULTY_ZH = {
    "easy": "简单",
    "medium": "中等",
    "hard": "困难",
}

NOTE_ZH = {
    "migration_doorkey_canonical": "规则不变：拿钥匙，开黄门，到达红色目标。",
    "migration_doorkey_medium": "规则不变：目标更远，并加入少量视觉干扰。",
    "migration_doorkey_hard": "规则不变：事件跨度更长，门后路径更远，干扰更多。",
    "migration_multiroom_canonical": "规则不变：穿过两扇顺序门，最终到达目标。",
    "migration_multiroom_medium": "规则不变：走廊更长，两次开门之间的间隔更大。",
    "migration_multiroom_hard": "规则不变：链路更长，目标更远，并加入额外干扰。",
    "migration_redblue_canonical": "规则不变：必须先开红门，再开蓝门。",
    "migration_redblue_medium": "规则不变：红门与蓝门之间的间隔更大。",
    "migration_redblue_hard": "规则不变：顺序约束相同，但时距更长、干扰更多。",
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


def fit_contain(img: Image.Image, target_w: int, target_h: int, *, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def load_summary(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"]
    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    rows.sort(key=lambda r: (r["family"], difficulty_rank.get(r["layout_metadata"].get("difficulty", ""), 9)))
    return rows


def zh_family(family: str) -> str:
    return FAMILY_ZH.get(family, family)


def zh_difficulty(difficulty: str) -> str:
    return DIFFICULTY_ZH.get(difficulty, difficulty)


def zh_note(row: dict[str, Any]) -> str:
    return NOTE_ZH.get(row["template_id"], row.get("note", ""))


def build_cover(rows: list[dict[str, Any]], mode: str, summary_path: Path) -> Image.Image:
    page, draw = create_blank_page()
    title_font = load_font(72)
    sub_font = load_font(42)
    body_font = load_font(34)

    draw.text((MARGIN, 120), "MiniWorld Task D 迁移环境试卷", font=title_font, fill=(0, 0, 0))
    mode_label = "盲审版" if mode == "blind" else "答案版"
    draw.text((MARGIN, 230), mode_label, font=sub_font, fill=(0, 0, 0))

    y = 340
    lines = [
        f"数据索引：{summary_path}",
        f"题目总数：{len(rows)}",
        "",
        "本试卷按 MiniGrid -> MiniWorld 的三类迁移环境组织：",
        "DoorKey、MultiRoom、RedBlue；每类环境均包含 简单 / 中等 / 困难 三档模板。",
        "",
        "盲审版说明：",
        "每页仅展示一条 full storyboard 轨迹与中文题面，不显示答案。",
        "",
        "答案版说明：",
        "每页展示 full storyboard、cf storyboard、事件覆盖图与关键元数据。",
    ]
    y = add_wrapped_text(draw, MARGIN, y, "\n".join(lines), body_font, PAGE_W - 2 * MARGIN, 48)

    y += 30
    for family in ("doorkey", "multiroom", "redblue"):
        family_rows = [r for r in rows if r["family"] == family]
        templates = ", ".join(
            f"{zh_difficulty(str(r['layout_metadata'].get('difficulty', '')))} = {r['template_id']}"
            for r in family_rows
        )
        y = add_wrapped_text(draw, MARGIN, y, f"{zh_family(family)}：{templates}", body_font, PAGE_W - 2 * MARGIN, 44)
        y += 12

    return page


def build_prompt_block(row: dict[str, Any], *, blind: bool) -> str:
    family = row["family"]
    difficulty = row["layout_metadata"].get("difficulty", "unknown")
    note = zh_note(row)
    seq_rel = row["manifest_relpath"]
    prompt = [
        f"环境家族：{zh_family(family)}",
        f"难度：{zh_difficulty(str(difficulty))}",
        f"模板名：{row['template_id']}",
        f"说明：{note}",
        "",
        "题目：",
        "判断这条 storyboard 连续轨迹是否成功完成任务。",
        "作答要求：只输出一个英文单词：SUCCESS 或 FAIL。",
    ]
    if not blind:
        prompt.extend(
            [
                "",
                f"full 是否成功 = {row['full_success']}",
                f"cf 是否成功 = {row['cf_success']}",
                f"cf 干预方式 = {INTERVENTION_ZH.get(row['cf_intervention'], row['cf_intervention'])}",
                f"full 事件步 = {row['full_event_steps']}",
                f"cf 事件步 = {row['cf_event_steps']}",
                f"manifest 路径 = {seq_rel}",
            ]
        )
    return "\n".join(prompt)


def build_row_page(row: dict[str, Any], *, root: Path, mode: str) -> Image.Image:
    blind = mode == "blind"
    page, draw = create_blank_page()
    title_font = load_font(52)
    body_font = load_font(32)
    small_font = load_font(26)

    family = row["family"]
    difficulty = row["layout_metadata"].get("difficulty", "unknown")
    title = f"{zh_family(family)} / {zh_difficulty(str(difficulty))} / {row['template_id']}"
    draw.text((MARGIN, 70), title, font=title_font, fill=(0, 0, 0))

    prompt_x = MARGIN
    prompt_y = 155
    prompt_w = 620
    prompt_h = PAGE_H - 2 * MARGIN - 40
    draw.rounded_rectangle((prompt_x, prompt_y, prompt_x + prompt_w, prompt_y + prompt_h), radius=22, outline=(0, 0, 0), width=3)
    text = build_prompt_block(row, blind=blind)
    add_wrapped_text(draw, prompt_x + 24, prompt_y + 24, text, body_font, prompt_w - 48, 42)

    right_x = prompt_x + prompt_w + 40
    right_w = PAGE_W - right_x - MARGIN

    full_img = Image.open(root / row["full_storyboard_relpath"]).convert("RGB")
    full_target_h = 1040 if blind else 760
    full_fit = fit_contain(full_img, right_w, full_target_h)
    page.paste(full_fit, (right_x, 150))
    draw.text((right_x, 118), "Full 轨迹图", font=body_font, fill=(0, 0, 0))

    if blind:
        meta = f"环境={zh_family(family)} | 难度={zh_difficulty(str(difficulty))} | 答案隐藏"
        draw.text((right_x, 1220), meta, font=small_font, fill=(70, 70, 70))
        return page

    cf_img = Image.open(root / row["cf_storyboard_relpath"]).convert("RGB")
    cf_fit = fit_contain(cf_img, right_w // 2 - 20, 620)
    event_cov_path = row.get("event_coverage_relpath")
    event_fit = None
    if event_cov_path:
        event_img = Image.open(root / event_cov_path).convert("RGB")
        event_fit = fit_contain(event_img, right_w // 2 - 20, 620)
    page.paste(cf_fit, (right_x, 980))
    draw.text((right_x, 948), "CF 轨迹图", font=body_font, fill=(0, 0, 0))
    if event_fit is not None:
        x2 = right_x + right_w // 2 + 20
        page.paste(event_fit, (x2, 980))
        draw.text((x2, 948), "关键事件覆盖图", font=body_font, fill=(0, 0, 0))

    meta_lines = [
        f"配对是否合格 = {row['pair_ok']}",
        f"CF 干预方式 = {INTERVENTION_ZH.get(row['cf_intervention'], row['cf_intervention'])}",
        f"full 事件步 = {row['full_event_steps']}",
        f"cf 事件步 = {row['cf_event_steps']}",
    ]
    add_wrapped_text(draw, right_x, 1660, "\n".join(meta_lines), small_font, right_w, 36, fill=(70, 70, 70))
    return page


def save_pdf(pages: list[Image.Image], out_pdf: Path) -> None:
    rgb_pages = [p.convert("RGB") for p in pages]
    rgb_pages[0].save(out_pdf, save_all=True, append_images=rgb_pages[1:], resolution=150.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render MiniWorld migration family showcase PDFs")
    ap.add_argument(
        "--summary_json",
        default="tmp_miniworld/taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json",
    )
    ap.add_argument(
        "--mode",
        choices=["blind", "answer-key"],
        default="answer-key",
    )
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
        out_pdf = root / f"taskD_MW_migration_showcase_{suffix}.pdf"
    else:
        out_pdf = Path(args.out_pdf)
    save_pdf(pages, out_pdf)
    print(f"Saved PDF to {out_pdf}")


if __name__ == "__main__":
    main()
